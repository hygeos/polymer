#!/usr/bin/env python
# encoding: utf-8

'''
Several tools for look-up tables management and interpolation

Provides:
    - LUT class: Look-Up Tables
      extends ndarrays for generic look-up table management including dimensions description,
      binary operations, multi-dimensional interpolation, plotting, etc
    - Idx class: find the index of values, for LUT interpolation
    - MLUT class: Multiple Look-Up Tables
      groups several LUTs, provides I/O interfaces with multiple data types
    - merge: MLUT merging
'''

from __future__ import print_function, division, absolute_import
import sys
import numpy as np
from scipy.interpolate import interp1d
from os.path import exists
from os import remove
from collections import OrderedDict
import warnings
if sys.version_info[:2] >= (3, 0): # python2/3 compatibility
    unicode = str
    xrange = range


def interleave_seq(p, q):
    '''
    Interleave 2 sequences (union, preserve order)
    ([1, 3, 4, 6], [2, 3, 6]) -> [1, 2, 3, 4, 6]
    '''
    if len(p) == 0:
        return q
    elif len(q) == 0:
        return p
    elif p[0] == q[0]:
        return [p[0]] + interleave_seq(p[1:], q[1:])
    elif p[0] in q[1:]:
        if q[0] in p[1:]:
            raise ValueError('sequences "{}" and "{}" cannot be interleaved'.format(p, q))
        return interleave_seq(q, p)
    else:  # p[0] not in q
        return [p[0]] + interleave_seq(p[1:], q)

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def uniq(seq):
    '''
    Returns uniques elements from a sequence, whist preserving its order
    http://stackoverflow.com/questions/480214/
    '''
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if not (x in seen or seen_add(x))]


def bin_edges(x):
    '''
    calculate n+1 bin edges from n bin centers in x
    '''
    assert x.ndim == 1
    if len(x) == 1:
        return np.array([x-0.5, x+0.5])
    else:
        first = (3*x[0] - x[1])/2.
        last = (3*x[-1] - x[-2])/2.
        return np.append(np.append(first, 0.5*(x[1:]+x[:-1])), last)


class LUT(object):
    '''
    Look-up table storage with generic multi-dimensional interpolation.
    Extends the __getitem__ method of ndarrays to float and float arrays (index
    tables with floats)
    The LUT axes can be optionally provided so that values can be interpolated
    into float indices in a first step, using the Idx class.
    Other features:
        * binary operations between LUTs and with scalars
        * automatic broadcasting:
          binary operations between LUTs with different axes
          the LUTs are broadcasted together according to their common axes
        * apply functions, dimension reduction (apply a function along a given axis)
        * equality testing
        * plotting

    Arguments:
        * data is a n-dimension array containing the LUT data
        * axes is a list representing each dimension, containing for each
          dimension:
            - a 1-d array or list containing the tabulated values of the dimension
            - or None, if there are no values associated with this dimension
        * names is an optional list of names for each axis (default: None)
          it is necessary for storing the LUT
        * attr is a dictionary of additional attributes, useful for merging LUTs
        * desc is a string describing the parameter stored (default None)
          when using save(), desc is used as the hdf dataset name to store the LUT

    Attributes: axes, shape, data, ndim, attrs, names, desc

    Example 1
    ---------
    Basic usage, without axes and in 1D:
    >>> data = np.arange(10)**2
    >>> L = LUT(data)
    >>> L[1], L[-1], L[::2]    # standard indexing is possible
    (1, 81, array([ 0,  4, 16, 36, 64]))
    >>> L[1.5]    # Indexing with a float: interpolation
    2.5
    >>> L[np.array([[0.5, 1.5], [2.5, 9.]])]    # interpolate several values at once
    array([[  0.5,   2.5],
           [  6.5,  81. ]])

    Example 2
    ---------
    Interpolation of the atmospheric pressure
    >>> z = np.linspace(0, 120., 80)
    >>> P0 = np.linspace(980, 1030, 6)
    >>> Pdata = P0.reshape(1,-1)*np.exp(-z.reshape(-1,1)/8) # dimensions (z, P0)

    A 2D LUT with attached axes.  Axes names can optionally be provided.
    >>> P = LUT(Pdata, axes=[z, P0], names=['z', 'P0'])
    >>> P[Idx(8.848), Idx(1013.)]  # standard pressure at mount Everest
    336.09126751112842
    >>> z = np.random.rand(50, 50)  # now z is a 2D array of elevations between 0 and 1 km
    >>> P0 = 1000+20*np.random.rand(50, 50)  # and P0 is a 2D array of pressures at z=0
    >>> P[Idx(z), Idx(P0)].shape    # returns a 2D array of corresponding pressures
    (50, 50)

    In Idx, the values can optionally be passed using keyword notation.
    In this case, there is a verification that the argument corresponds to the right axis name.
    >>> P[:, Idx(1013., 'P0')].shape   # returns the (shape of) standard vertical pressure profile
    (80,)
    '''

    def __init__(self, data, axes=None, names=None, desc=None, attrs=None):
        self.data = data
        self.desc = desc
        if attrs is None:
            self.attrs = OrderedDict()
        else:
            self.attrs = attrs
        self.ndim = len(self.data.shape)
        self.shape = data.shape

        # check axes
        if axes is None:
            self.axes = self.ndim * [None]
        else:
            self.axes = axes
            assert len(axes) == self.ndim
            for i, ax in enumerate(axes):
                if isinstance(ax, np.ndarray):
                    assert ax.ndim == 1
                    assert len(ax) == data.shape[i]
                elif isinstance(ax, list):
                    assert len(ax) == data.shape[i]
                elif ax is None: pass
                else:
                    assert ax[:].ndim == 1
                    assert len(ax) == data.shape[i]

        # check names
        if names is None:
            self.names = self.ndim * [None]
        else:
            self.names = names
            assert len(names) == self.ndim

        if np.issubdtype(self.data.dtype, np.float):
            self.formatter = '{:3g}'
        else:
            self.formatter = '{}'

    def sub(self, d=None, ignore=False):
        '''
        returns a subset LUT of current LUT along several axes

        * d is a dictionary of {ax: value}, where:

            - ax is the name (string) or index (integer) of the axis to consider

            - value can be:
                * scalar integer (axis is removed)
                * float index (axis is removed with interpolation)
                * slice (axis is subsetted)
                * 1-d array (axis is subsetted)
                * Idx

        * ignore: if True, return full LUT if axis is not present in LUT

        Examples:
          lut.sub({'axis1': 1.5})
             returns the lut stripped from 'axis1', for which we use the index 1.5
          lut.sub({2: Idx(42.)})
          lut.sub({'ax': arange(5)})   # first 5 values
          lut.sub({'z': slice(None,None,2)})   # every other value
          lut.sub({'wav': lut.axis('wav')<500.})   # array of booleans
          lut.sub({'wav': Idx(lambda x: x<500.)})  # select wav < 500
        '''
        if d is None:
            return Subsetter(self)

        if self.ndim == 0:
            # scalar lut, cannot subset
            if ignore:
                return self
            else:
                raise Exception('Cannot subset scalar LUT {}'.format(self))

        keys = [slice(None)] * self.ndim
        names = list(self.names)
        axes = list(self.axes)
        dims_to_remove = []

        for ax, v in d.items():
            # ax: str or int
            if isinstance(ax, int):
                iax = ax
            else:
                assert isinstance(ax, str), 'ax should be str or int'
                if ax in self.names:
                    iax = self.names.index(ax)
                else:
                    if ignore:
                        continue
                    else:
                        raise Exception('sub is not possible on axis "{}" '
                                        'because this axis is not present '
                                        'in {}'.format(ax, self))

            if isinstance(v, Idx_base):
                idx = v.index(self.axes[iax])
                newax = v.apply(self.axes[iax])
            else:
                idx = v
                newax = None

            # axes: names and values
            if (np.array(idx).ndim == 0) and not isinstance(v, slice):
                # scalar, remove dimension
                dims_to_remove.append(iax)
            # in all other cases, keep dimension
            elif axes[iax] is None:
                axes[iax] = None
            elif isinstance(v, slice):
                axes[iax] = axes[iax][v]
            elif np.array(idx).ndim == 1:
                if newax is not None:
                    axes[iax] = newax
                elif v.dtype.kind in ['u', 'i']: # integer
                    axes[iax] = axes[iax][v]
                else:
                    axes[iax] = None
            else:
                raise Exception('Cannot use a {}-dim array to '
                                'subset a LUT'.format(np.array(idx).ndim))

            keys[iax] = idx

        axes = [a for i, a in enumerate(axes) if not i in dims_to_remove]
        names = [a for i, a in enumerate(names) if not i in dims_to_remove]

        data = self[tuple(keys)]

        return LUT(data, axes=axes, names=names,
                   attrs=dict(self.attrs), desc=self.desc)


    def axis(self, a, aslut=False):
        '''
        returns axis referred to by a (string or integer)
        aslut:
            False: returns the values
            True: returns axis a a LUT
                (containing itself as only axis)
        '''
        if isinstance(a, str):
            index = self.names.index(a)
        elif isinstance(a, int):
            index = a
        else:
            raise TypeError('argument of LUT.axis() should be int or string')

        if aslut:
            data = self.axes[index]
            return LUT(data, axes=[data], names=[self.names[index]])
        else:
            return self.axes[index]

    def print_info(self, *args, **kwargs):
        # same as describe()
        return self.describe(*args, **kwargs)

    def describe(self, show_attrs=False):
        '''
        Prints the LUT informations
        arguments:
            show_attrs: show LUT attributes (default: False)

        returns self
        '''
        try:
            rng = ' between {:.3g} and {:.3g}'.format(
                    np.amin(self.data), np.amax(self.data)
                    )
        except:
            rng = ''

        print('LUT {}({}{}):'.format(
                {True: '', False: '"{}" '.format(self.desc)}[self.desc is None],
                self.data.dtype, rng,
                ))

        for i in xrange(len(self.data.shape)):
            if self.names[i] is None:
                name = 'NoName'
            else:
                name = self.names[i]
            if self.axes[i] is None:
                print('  Dim {} ({}): {} values, no axis attached'.format(i, name, self.data.shape[i]))
            else:
                print('  Dim {} ({}): {} values in [{}, {}]'.format(
                        i, name,
                        len(self.axes[i]),
                        self.axes[i][0],
                        self.axes[i][-1]))
        if show_attrs:
            print(' Attributes:')
            for k, v in self.attrs.items():
                print(' ', k, ':', v)

        return self

    def __getitem__(self, keys):
        '''
        Get items from the LUT, with possible interpolation

        Indexing works mostly like a standard array indexing, with the following differences:
            - float indexes can be provided, they will result in an
              interpolation between bracketing integer indices for this dimension
            - float arrays can be provided, they will result in an
              interpolation between bracketing integer arrays for this dimension
            - Idx objects can be provided, they are first converted into indices
            - basic indexing and slicing, and advanced indexing (indexing with
              an ndarray of int or bool) can still be used
              (see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)
            - if arrays are passed as arguments, they must all have the same shape
            - Unlike ndarrays, the number of dimensions in keys should be
              identical to the dimension of the LUT
              >>> LUT(np.zeros((2, 2)))[:]
              Traceback (most recent call last):
              ...
              Exception: Incorrect number of dimensions in __getitem__

        Returns: a scalar or ndarray
        '''

        if self.data.dtype.char in ['S', 'U']:
            # string or unicode arrays: bypass this method and directly apply
            # ndarray.__getitem__
            return self.data[keys]

        if not isinstance(keys, tuple):
            keys = (keys,)
        keys = list(keys)
        N = len(keys)

        if N != self.ndim:
            raise Exception('Incorrect number of dimensions in __getitem__ '
                            '(expecting {}, got {})'.format(self.ndim, N))

        # convert the Idx keys to float indices
        for i in xrange(N):
            k = keys[i]
            if isinstance(k, Idx_base):
                # k is an Idx instance
                # convert it to float indices for the current axis
                if k.name not in [None, self.names[i]]:
                    msg = 'Error, wrong parameter passed at position {}, expected {}, got {}'
                    raise Exception(msg.format(i, self.names[i], k.name))
                keys[i] = k.index(self.axes[i])

        # determine the dimensions of the result (for broadcasting coef)
        dims_array = None
        index0 = []
        for i in xrange(N):
            k = keys[i]
            if isinstance(k, np.ndarray) and (k.ndim > 0):
                index0.append(np.zeros_like(k, dtype='int'))
                if dims_array is None:
                    dims_array = k.shape
                else:
                    assert dims_array == k.shape, 'LUTS.__getitem__: all arrays must have same shape ({} != {})'.format(str(dims_array), str(k.shape))
            elif isinstance(k, slice):
                index0.append(k)
            else:  # scalar
                index0.append(0)

        shp_res = np.zeros(1).reshape([1]*self.ndim)[index0].shape

        # determine the interpolation axes
        # and for those axes, determine the lower index (inf) and the weight
        # (x) between lower and upper index
        interpolate_axis = []   # indices of the interpolated axes
        inf_list = []       # index of the lower elements for the interpolated axes
        x_list = []         # weights, for the interpolated axes
        for i in xrange(N):
            k = keys[i]

            # floating-point indices should be interpolated
            interpolate = False
            if isinstance(k, np.ndarray) and (k.dtype in [np.dtype('float32'),
                                                          np.dtype('float64')]):
                interpolate = True
                inf = k.astype('int')
                inf[inf == self.data.shape[i]-1] -= 1
                x = k-inf
                if k.ndim > 0:
                    x = x.reshape(shp_res)
            elif isinstance(k, float):
                interpolate = True
                inf = int(k)
                if inf == self.data.shape[i]-1:
                    inf -= 1
                x = k-inf
            if interpolate:
                # current axis needs interpolation
                inf_list.append(inf)
                x_list.append(x)
                interpolate_axis.append(i)

        # loop over the 2^n bracketing elements
        # (cartesian product of [0, 1] over n dimensions)
        n = len(interpolate_axis)
        result = 0
        for b in xrange(2**n):

            # coefficient attributed to the current item
            # and adjust the indices
            # for the interpolated dimensions
            coef = 1
            for i in xrange(n):
                # bb is the ith bit in b (0 or 1)
                bb = ((1<<i)&b)>>i
                x = x_list[i]
                if bb:
                    coef *= x
                else:
                    coef *= 1-x

                keys[interpolate_axis[i]] = inf_list[i] + bb

            result += coef * self.data[tuple(keys)]

        return result


    def equal(self, other, strict=True):
        '''
        Checks equality between two LUTs:
            - same axes
            - same shape
            - same values (if strict)
        '''
        if not isinstance(other, LUT):
            return False
        for i, ax in enumerate(self.axes):
            if (ax is None) and (other.axes[i] is None):
                continue
            if (ax is None) or (other.axes[i] is None):
                return False
            if not np.allclose(ax, other.axes[i]):
                return False
        if not self.data.shape == other.data.shape:
            return False
        if strict:
            if not np.allclose(self.data, other.data):
                return False
            if self.attrs != other.attrs:
                return False

        return True


    def __binary_operation_lut__(self, other, fn):
        '''
        apply fn(self, other) where other is a LUT
        the result is determined by using common axes between self and other
        and using appropriate broadcasting
        '''
        # shapes for broadcasting self vs other
        # None adds an singleton dimension, slice(None) adds a full dimension
        shp1, shp2 = [], []

        # new axes
        axes = []

        # determine union of axes
        names = interleave_seq(self.names, other.names)
        for i, a in enumerate(names):
            if a in self.names:
                axes.append(self.axes[self.names.index(a)])
                shp1.append(slice(None))
                if a in other.names:
                    shp2.append(slice(None))
                else:
                    shp2.append(None)
            else:
                axes.append(other.axes[other.names.index(a)])
                shp1.append(None)
                shp2.append(slice(None))

        # include common attributes
        attrs = {}
        for k in self.attrs:
            # check that the attributes are equal
            if not (k in other.attrs):
                continue
            if isinstance(self.attrs[k], np.ndarray):
                if not isinstance(other.attrs[k], np.ndarray):
                    continue
                if not np.allclose(self.attrs[k], other.attrs[k]):
                    continue
            else:
                if self.attrs[k] != other.attrs[k]:
                    continue
            attrs.update({k: self.attrs[k]})

        if self.desc == other.desc:
            desc = self.desc
        else:
            desc = str(fn)

        return LUT(
                fn(self.data[tuple(shp1)],
                    other.data[tuple(shp2)]),
                axes=axes, names=names,
                attrs=attrs, desc=desc)


    def __binary_operation_scalar__(self, other, fn):
        return LUT(fn(self.data, other),
                axes=self.axes, names=self.names,
                attrs=self.attrs, desc=self.desc)

    def __binary_operation__(self, other, fn):
        if isinstance(other, LUT):
            return self.__binary_operation_lut__(other, fn)
        else:
            return self.__binary_operation_scalar__(other, fn)

    def __add__(self, other):
        return self.__binary_operation__(other, lambda x, y: x+y)

    def __radd__(self, other):
        return self.__binary_operation__(other, lambda x, y: x+y)

    def __sub__(self, other):
        return self.__binary_operation__(other, lambda x, y: x-y)

    def __rsub__(self, other):
        return self.__binary_operation__(other, lambda x, y: y-x)

    def __mul__(self, other):
        return self.__binary_operation__(other, lambda x, y: x*y)

    def __rmul__(self, other):
        return self.__binary_operation__(other, lambda x, y: x*y)

    def __div__(self, other):
        return self.__binary_operation__(other, lambda x, y: x/y)

    def __rdiv__(self, other):
        return self.__binary_operation__(other, lambda x, y: y/x)

    def __truediv__(self, other):
        return self.__binary_operation__(other, lambda x, y: x/y)

    def __rtruediv__(self, other):
        return self.__binary_operation__(other, lambda x, y: y/x)

    def __eq__(self, other):
        return self.equal(other)

    def __neq__(self, other):
        return not self.equal(other)


    def to_mlut(self):
        '''
        convert to a MLUT
        '''
        m = MLUT()

        # axes
        if self.axes is not None:
            for i in xrange(len(self.axes)):
                name = self.names[i]
                axis = self.axes[i]
                if (name is None) or (axis is None):
                    continue
                m.add_axis(name, axis)

        # datasets
        m.add_dataset(self.desc, self.data, axnames=self.names, attrs=self.attrs)

        # attributes
        m.set_attrs(self.attrs)

        return m

    def apply(self, fn, desc=None):
        '''
        returns a LUT whose content is obtained by applying function fn
        if desc is provided, use this description
        '''
        if (desc is None) and (self.desc is not None):
            desc = self.desc
        return LUT(fn(self.data),
                axes=self.axes, names=self.names,
                attrs=self.attrs, desc=desc)

    def reduce(self, fn, axis, grouping=None, as_lut=False, **kwargs):
        '''
        apply function fn to a given axis
        fn: function to apply
            should be applicable to a numpy.ndarray and support argument axis
            (example: numpy.sum)
        axis: name (str) or index of axis
        grouping: iterable of same size as axis
                  fn is applied by groups corresponding to identical values in
                  grouping
                      example: grouping = [0, 0, 0, 1, 1, 2]
                      results in fn(3 first elements), fn(2 next), then fn(last)
                    the axis of the reduced axis takes the values of grouping
                  default None (apply to all elements, remove axis)
        as_lut: if axis reduction results in a scalar, returns a dimensionless LUT
                default False (returns a scalar)

        '''
        if isinstance(axis, str):
            index = self.names.index(axis)
        else:
            index = axis

        if grouping is None:
            axes = list(self.axes)
            names = list(self.names)
            axes.pop(index)
            names.pop(index)
            if (self.ndim == 1) and (not as_lut):
                # returns a scalar
                return fn(self.data, axis=index, **kwargs)
            else:
                # returns a LUT
                return LUT(fn(self.data, axis=index, **kwargs),
                        axes=axes, names=names,
                        attrs=self.attrs, desc=self.desc)
        else:
            assert len(grouping) == len(self.axes[index])
            shp = list(self.data.shape)
            U = uniq(grouping)
            shp[index] = len(U)
            data = np.zeros(shp, dtype=self.data.dtype)
            ind1 = [slice(None),] * self.ndim
            ind2 = [slice(None),] * self.ndim
            for i, u in enumerate(U):
                # fill each group
                ind1[index] = i
                ind2[index] = (grouping == u)
                data[tuple(ind1)] = fn(self.data[tuple(ind2)], axis=index, **kwargs)
            axes = list(self.axes)
            axes[index] = U
            return LUT(data,
                    axes=axes, names=self.names,
                    attrs=self.attrs, desc=self.desc)


    def swapaxes(self, axis1, axis2):
        '''
        Swaps two axes of the LUT

        Arguments:
            axis1 and axis2 (int or str): the name or index of the axes to swap

        Returns: the swapped LUT
        '''
        if isinstance(axis1, str):
            axis1 = self.names.index(axis1)
        if isinstance(axis2, str):
            axis2 = self.names.index(axis2)

        # swap the names and axes
        names, axes = [], []
        for i in xrange(self.ndim):
            names.append(self.names[i])
            axes.append(self.axes[i])
        names[axis1], names[axis2] = names[axis2], names[axis1]
        axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

        return LUT(self.data.swapaxes(axis1, axis2),
                   axes=axes, names=names,
                   attrs=self.attrs, desc=self.desc)


    def plot(self, *args, **kwargs):
        '''
        Plot a 1 or 2-dimension LUT
        Note:
        in 2-dim, creates a new figure

        Arguments:
            * swap: swap x and y axes
            * show_grid
            * fmt: format for plotting 1d data

            Only in 2-dim:
            * vmin, vmax: color scale extent (default: from min/max)
            * cmap: colormap instance
            * index: index (or Idx instance) for plotting transect
        '''
        if self.ndim == 0:
            if self.desc is None:
                print(self.formatter.format(self.data))
            else:
                print(('{}: '+self.formatter).format(self.desc, self.data))
        elif self.ndim == 1:
            self.__plot_1d(*args, **kwargs)
        elif self.ndim == 2:
            self.__plot_2d(*args, **kwargs)
        else:
            self.plot_nd(*args, **kwargs)

        return self


    def __plot_1d(self, show_grid=True, swap=False, fmt=None, label=None, 
                  vmin=None, vmax=None, legend=False, plot=None, **kwargs):
        '''
        plot a 1-dimension LUT, returns self

        arguments:
            plot: custom plot function (defaults to plot)
                  example: plot=semilogy
        '''
        from pylab import xlabel, ylabel, grid, ylim, ticklabel_format
        import pylab as pl

        if plot is None:
            plot = pl.plot

        # no plotting for string datasets
        if self.data.dtype.char == 'S':
            warnings.warn('1D plot does not work for string arrays')
            return

        ax = self.axes[0]
        if ax is None:
            ax = range(self.shape[0])

        if vmin is None:
            vmin = np.amin(self.data[~np.isnan(self.data)])
        if vmax is None:
            vmax = np.amax(self.data[~np.isnan(self.data)])

        if not swap:
            xx = ax
            yy = self.data
            xlab = self.names[0]
            ylab = self.desc
        else:
            xx = self.data
            yy = ax
            xlab = self.desc
            ylab = self.names[0]

        if fmt is None:
            plot(xx, yy, label=label)
        else:
            plot(xx, yy, fmt, label=label)
        ylim(vmin,vmax)
        if xlab is not None:
            xlabel(xlab)
        if ylab is not None and not legend:
            ylabel(ylab)

        if plot == pl.plot:
            ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

        grid(show_grid)
        if legend:
            leg = pl.legend(loc='best', fancybox=True)
            leg.get_frame().set_alpha(0.5)


    def __plot_2d(self, fmt='k-', show_grid=True, swap=False,
                  vmin=None, vmax=None, cmap=None, index=None,
                  label=None, **kwargs):
        '''
        Plot a 2-dimension LUT, with optional transect
        returns self
        '''
        import matplotlib.pyplot as plt

        if vmin is None:
            vmin = np.amin(self.data[~np.isnan(self.data)])
        if vmax is None:
            vmax = np.amax(self.data[~np.isnan(self.data)])

        if cmap is None:
            cmap = plt.cm.jet
            cmap.set_under('black')
            cmap.set_over('white')
            cmap.set_bad('0.5') # grey 50%

        if not swap:
            axis1, axis2 = self.axes[1], self.axes[0]
            lab1, lab2 = self.names[1], self.names[0]
            data = self.data
        else:
            axis1, axis2 = self.axes[0], self.axes[1]
            data = np.swapaxes(self.data, 0, 1)
            lab1, lab2 = self.names[0], self.names[1]

        if axis1 is None:
            axis1 = np.arange(data.shape[1])
        if axis2 is None:
            axis2 = np.arange(data.shape[0])

        if index is None:
            fig, ax1 = plt.subplots(nrows=1, ncols=1)
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

        axis1 = bin_edges(axis1)
        axis2 = bin_edges(axis2)

        X, Y = np.meshgrid(axis1, axis2)
        plt.sca(ax1)
        im = plt.pcolormesh(X, Y, data, vmin=vmin, vmax=vmax)
        plt.grid(show_grid)
        plt.axis([
            np.amin(axis1),
            np.amax(axis1),
            np.amin(axis2),
            np.amax(axis2),
            ])
        cbar_ax = fig.add_axes([0.92, 0.25, 0.03, 0.5])
        # cbar_ax.text(0, -0.1, self.desc, verticalalignment='top')
        if label is None:
            label = self.desc
        if label is not None:
            cbar_ax.set_title(self.desc, weight='bold', horizontalalignment='left', position=(0.,-0.15))
        plt.colorbar(im, extend='both', orientation='vertical', cax=cbar_ax)

        if index is None:
            plt.sca(ax1)
            plt.ylabel(lab2)
            plt.xlabel(lab1)
        else:
            # show transect
            if isinstance(index, Idx):
                v = index.value
            else:
                v = axis2[index]

            ax1.plot([np.amin(axis1), np.amax(axis1)], [v, v], 'w--')
            ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

            if not swap:
                ax2.plot(axis1, self[index, :], fmt)
            else:
                ax2.plot(axis1, self[:, index], fmt)
            ax2.grid(show_grid)

            plt.sca(ax1)
            plt.ylabel(lab2)
            plt.sca(ax2)
            plt.ylabel(self.desc)
            plt.xlabel(lab1)

            ax2.axis(ymin=vmin, ymax=vmax)
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

        fig.subplots_adjust(hspace=0.)

    def plot_nd(self, *args, **kwargs):
        try:
            from ipywidgets import VBox, HBox, Checkbox, IntSlider, HTML
            from IPython.display import display, clear_output
        except ImportError:
            raise Exception('IPython notebook widgets are required '
                    'to plot a LUT with more than 2 dimensions')
        wid = []
        chks, sliders, texts = [], [], []
        for i in xrange(self.ndim):
            name = self.names[i]
            ax = self.axes[i]
            desc = name
            if desc is None:
                desc = 'NoName'
            if ax is not None:
                desc += ' [{:.5g}, {:.5g}]'.format(ax[0], ax[-1])
            chk = Checkbox(description=desc, value=False)
            slider = IntSlider(min=0, max=self.shape[i]-1)
            slider.visible = False
            text = HTML(value='')
            text.visible = False
            chks.append(chk)
            sliders.append(slider)
            texts.append(text)
            wid.append(HBox([chk, text, slider]))

        def update():
            clear_output()

            keys = []
            ndim = 0
            for i in xrange(self.ndim):
                # set sliders visibility
                sliders[i].visible = chks[i].value
                texts[i].visible = chks[i].value

                if chks[i].value:
                    index = sliders[i].value
                    keys.append(index)
                    # set text (value)
                    if self.axes[i] != None:
                        texts[i].value = '&nbsp;&nbsp;{:.5g}&nbsp;&nbsp;'.format(self.axes[i][index])
                else:
                    keys.append(slice(None))
                    ndim += 1

            if ndim <= 2:
                self.sub().__getitem__(tuple(keys)).plot(*args, **kwargs)
            else:
                print('Please select at least {} dimension(s)'.format(ndim-2))

        for i in xrange(self.ndim):
            chks[i].on_trait_change(update, 'value')
            sliders[i].on_trait_change(update, 'value')

        display(VBox(wid))
        update()

    def plot_polar(self, *args, **kwargs):
        plot_polar(self, *args, **kwargs)
        return self

    def plot_semi(self, *args, **kwargs):
        kwargs['semi'] = True
        plot_polar(self, *args, **kwargs)
        return self

    def transect2D(self, *args, **kwargs):
        transect2D(self, *args, **kwargs)
        return self


def Idx(value, name=None, round=False, fill_value=None):
    '''
    Calculate the indices of values by interpolation in a LUT axis
    The index method is typically called when indexing a dimension of a LUT
    object by a Idx object.
    The round attribute (boolean) indicates whether the resulting index should
    be rounded to the closest integer.

    Idx value can be of several kinds:
    - scalar (int or float)
     => index return scalar float index
    - array-like
      => index return array-like float index
    - callable (ex: lambda x: x<5)
      => index returns callable applied to axis

    Example: find the float index of 35. in an array [0, 10, ..., 100]
    >>> Idx(35.).index(np.linspace(0, 100, 11))
    array(3.5)

    Find the indices of several values in the array [0, 10, ..., 100]
    >>> Idx(np.array([32., 45., 72.])).index(np.linspace(0, 100, 11))
    array([ 3.2,  4.5,  7.2])

    Optionally, the name of the parameter can be provided as a keyword
    argument.
    Example: Idx(3., 'a') instead of Idx(3.)
    This allows verifying that the parameter is used in the right axis.

    Options:
        - fill_value are passed to interp1d (implies bounds_error=False)
          Special value for fill_value='extrema': fill with extrema values, don't extrapolate

    NOTE: this function is a factory that returns an Idx_* instance based on input type

    Method apply(axis):
        returns the Idx values applied to axis instead of their index
    '''
    if hasattr(value, '__call__'):
        return Idx_filter(value, name=name)
    else:
        return Idx_arr(value, name=name,
                       round=round, fill_value=fill_value)


class Idx_base(object):
    '''
    Base class for Idx objects
    '''
    def __init__(self, value, name=None, round=False, fill_value=None):
        if value is not None:
            self.value = value
            self.name = name
            self.round = round
            self.fill_value = fill_value


class Idx_arr(Idx_base):
    '''
    Idx class for array-like values
    '''
    def index(self, axis):
        '''
        Return the floating point index of the values in the axis
        '''
        if len(axis) == 1:
            if not np.allclose(np.array(self.value), axis[0]):
                raise ValueError("(Idx) Out of axis value (value={}, axis={})".format(self.value, axis))
            return 0

        else:
            # axis is scalar or ndarray: interpolate
            if self.fill_value == 'extrema':
                fv = (0, len(axis)-1)
            elif self.fill_value == 'extrema,warn':
                fv = (0, len(axis)-1)
                if (np.amax(self.value) > np.amax(axis)):
                    warnings.warn('(Idx) Value {} is above the axis maximum {} (axis {})'.format(
                        np.amax(self.value), np.amax(axis), self.name))
                if (np.amin(self.value) < np.amin(axis)):
                    warnings.warn('(Idx) Value {} is under the axis minimum {} (axis {})'.format(
                        np.amin(self.value), np.amin(axis), self.name))
            else:
                fv = self.fill_value
            be = (self.fill_value is None)

            res = interp1d(axis, np.arange(len(axis)),
                    bounds_error=be,
                    fill_value=fv)(self.value)
            if self.round:
                if isinstance(res, np.ndarray):
                    res = res.round().astype(int)
                else:
                    res = round(res)
            return res


    def apply(self, axis=None):
        return self.value

class Idx_filter(Idx_base):
    '''
    Idx class for filtering functions
    '''
    def index(self, axis):
        # value is a callable (function):
        # returns value applied to axis
        return self.value(axis)

    def apply(self, axis):
        return axis[self.index(axis)]


class Subsetter(object):
    '''
    A conveniency class to use the syntax like:
    LUT.sub()[:,:,0]
    for subsetting LUTs
    '''
    def __init__(self, LUT):
        self.LUT = LUT

    def __getitem__(self, keys):
        '''
        subset parent LUT
        '''
        return self.LUT.sub(dict(enumerate(keys)))


def plot_polar(lut, index=None, vmin=None, vmax=None, rect='211', sub='212',
               sym=True, swap='auto', fig=None, cmap=None, semi=False):
    '''
    Contour and eventually transect of 2D LUT on a semi polar plot, with
    dimensions (angle, radius)

    lut: 2D look-up table to display
            with axes (radius, angle) (unless swapped)
            angle is assumed to be in degrees and is not scaled
    index: index of the item to transect in the 'angle' dimension
           can be an Idx instance of several values or a list of indices
           if None (default), no transect
    vmin, vmax: range of values
                default None: determine min/max from values
    rect: subplot position of the main plot ('111' for example)
    sub: subplot position of the transect
    sym: the transect uses symmetrical axis (boolean)
         if None (default), use symmetry iff axis is 'zenith'
    swap: swap the order of the 2 axes to (radius, angle)
          if 'auto', searches for 'azi' in both axes names
    fig : destination figure. If None (default), create a new figure.
    cmap: color map
    semi: polar by default, otherwise semi polar if lut is computed for 360 deg
    '''
    from pylab import figure, cm
    import mpl_toolkits.axisartist.angle_helper as angle_helper
    from matplotlib.transforms import Affine2D
    from mpl_toolkits.axisartist import floating_axes
    from matplotlib.projections import PolarAxes

    #
    # initialization
    #
    Phimax = 360.
    if semi : Phimax=180.

    assert lut.ndim == 2

    show_sub = index is not None
    if fig is None:
        if show_sub:
            fig = figure(figsize=(4.5, 4.5))
        else:
            fig = figure(figsize=(4.5, 6))

    if swap =='auto':
        if ('azi' in lut.names[1].lower()) and ('azi' not in lut.names[0].lower()):
            swap = True
        else:
            swap = False

    # ax1 is angle, ax2 is radius
    if swap:
        ax1, ax2 = lut.axes[1], lut.axes[0]
        name1, name2 = lut.names[1], lut.names[0]
        data = np.swapaxes(lut.data, 0, 1)
    else:
        ax1, ax2 = lut.axes[0], lut.axes[1]
        name1, name2 = lut.names[0], lut.names[1]
        data = lut.data

    if vmin is None:
        vmin = np.amin(lut.data[~np.isnan(lut.data)])
    if vmax is None:
        vmax = np.amax(lut.data[~np.isnan(lut.data)])
    if vmin == vmax:
        vmin -= 0.001
        vmax += 0.001
    if vmin > vmax: vmin, vmax = vmax, vmin

    #
    # semi polar axis
    #
    # angle
    ax1_scaled = ax1
    label1 = name1

    # radius axis
    # scale to [0, 90]
    ax2_min = np.amin(ax2)
    ax2_max = np.amax(ax2)
    ax2_scaled = (ax2 - ax2_min)/(ax2_max- ax2_min)*90.
    label2 = name2

    # angle axis
    grid_locator1 = angle_helper.LocatorDMS({True: 4, False:8}[semi], include_last=False)
    tick_formatter1 = angle_helper.FormatterDMS()

    class Locator(object):
        def __call__(self, *args):
            # as returned by angle_helper.step
            return [np.array([0, 30, 60, 90]), 4, 1.0]
    class Formatter(object):
        def __call__(self, *args):
            return map(lambda x: '{:.3g}'.format(x), np.linspace(ax2_min, ax2_max, 4))

    # radius axis
    if ((ax2_min < 10.) and (ax2_min >= 0)
            and (ax2_max <= 90) and (ax2_max > 80)):
        # angle in degrees
        grid_locator2 = angle_helper.LocatorDMS(4)
        tick_formatter2 = angle_helper.FormatterDMS()
    else:
        # custom locator
        grid_locator2 = Locator()
        tick_formatter2 = Formatter()

    tr_rotate = Affine2D().translate(0, 0)  # orientation
    tr_scale = Affine2D().scale(np.pi/180., 1.)  # scale to radians

    tr = tr_rotate + tr_scale + PolarAxes.PolarTransform()

    grid_helper = floating_axes.GridHelperCurveLinear(tr,
                                    extremes=(0., Phimax, 0., 90.),
                                    grid_locator1=grid_locator1,
                                    grid_locator2=grid_locator2,
                                    tick_formatter1=tick_formatter1,
                                    tick_formatter2=tick_formatter2,
                            )

    ax_polar = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    fig.add_subplot(ax_polar)

    # adjust axis
    ax_polar.grid(True)
    ax_polar.axis["left"].set_axis_direction("bottom")
    ax_polar.axis["right"].set_axis_direction("top")
    ax_polar.axis["bottom"].set_visible(False)
    ax_polar.axis["top"].set_axis_direction("bottom")
    ax_polar.axis["top"].toggle(ticklabels=True, label=True)
    ax_polar.axis["top"].major_ticklabels.set_axis_direction("top")
    ax_polar.axis["top"].label.set_axis_direction("top")

    #ax_polar.axis["top"].axes.text(0.70, 0.92, label1,
    ax_polar.axis["top"].axes.text(0.72, 0.98, label1,
                                   transform=ax_polar.transAxes,
                                   ha='left',
                                   va='bottom')
    #ax_polar.axis["left"].axes.text(0.25, -0.03, label2,
    ax_polar.axis["left"].axes.text(0.10, -0.03, label2,
                                   transform=ax_polar.transAxes,
                                   ha='center',
                                   va='top')

    # create a parasite axes whose transData in RA, cz
    aux_ax_polar = ax_polar.get_aux_axes(tr)

    aux_ax_polar.patch = ax_polar.patch # for aux_ax to have a clip path as in ax
    ax_polar.patch.zorder=0.9 # but this has a side effect that the patch is
                        # drawn twice, and possibly over some other
                        # artists. So, we decrease the zorder a bit to
                        # prevent this.

    #
    # initialize the cartesian axis below the semipolar
    #
    if show_sub:
        ax_cart = fig.add_subplot(sub)
        if sym:
            ax_cart.set_xlim(-ax2_max, ax2_max)
        else:
            ax_cart.set_xlim(ax2_min, ax2_max)
        ax_cart.set_ylim(vmin, vmax)
        ax_cart.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax_cart.grid(True)

    #
    # draw colormesh
    #
    if cmap is None:
        cmap = cm.rainbow
        cmap.set_under('black')
        cmap.set_over('white')
        cmap.set_bad('0.5') # grey 50%
    r, t = np.meshgrid(bin_edges(ax2_scaled), bin_edges(ax1_scaled))
    masked_data = np.ma.masked_where(np.isnan(data) | np.isinf(data), data)
    im = aux_ax_polar.pcolormesh(t, r, masked_data, cmap=cmap, vmin=vmin, vmax=vmax)

    if show_sub:
        # convert Idx instance to index if necessarry
        if isinstance(index, Idx_base):
            indexes = np.round(index.index(ax1)).astype(int)
            #indexes = int(round(index.index(ax1)))
        else: indexes = index
        for ii,index in enumerate(indexes):
            if semi:
                mirror_index = -1 -index
            else:
                mirror_index = (ax1_scaled.shape[0]//2 + index)%ax1_scaled.shape[0]
            # draw line over colormesh
            vertex0 = np.array([[0,0],[ax1_scaled[index],ax2_max]])
            vertex1 = np.array([[0,0],[ax1_scaled[mirror_index],ax2_max]])
            aux_ax_polar.plot(vertex0[:,0],vertex0[:,1], 'w')
            if sym:
                aux_ax_polar.plot(vertex1[:,0],vertex1[:,1],'w--',linewidth=2)

            #
            # plot transects
            #
            color = ['k', 'r','g','b','m','y'][ii%6]
            ax_cart.plot(ax2, data[index,:],'-'+color)
            if sym:
                ax_cart.plot(-ax2, data[mirror_index,:],'--'+color)

    # add colorbar
    fig.colorbar(im, orientation='horizontal', extend='both', ticks=np.linspace(vmin, vmax, 5))
    if lut.desc is not None:
        ax_polar.set_title(lut.desc, weight='bold', position=(0.05,0.97))


def transect2D(lut, index=None, vmin=None, vmax=None, sym=True, swap='auto', fig=None, sub=121, color='k', percent=False, fmt='-'):
    '''
    Transect of 2D LUT

    lut: 2D look-up table to display
            with axes (radius, angle) (unless swapped)
            angle is assumed to be in degrees and is not scaled
    index: index of the item to transect in the 'angle' dimension
           can be an Idx instance
           if None (default), no transect
    vmin, vmax: range of values
                default None: determine min/max from values
    sym: the transect uses symmetrical axis (boolean)
         if None (default), use symmetry if axis is 'zenith'
    swap: swap the order of the 2 axes to (radius, angle)
          if 'auto', searches for 'azi' in both axes names
    fig : destination figure. If None (default), create a new figure.
    color : color of the transect
    percent: if True set scale to 0 to 100%
    '''
    from pylab import figure

    assert lut.ndim == 2

    if fig is None:
        fig = figure(figsize=(4.5, 2.5))

    if swap =='auto':
        if ('azi' in lut.names[1].lower()) and ('azi' not in lut.names[0].lower()):
            swap = True
        else:
            swap = False

    # ax1 is angle, ax2 is radius
    if swap:
        ax1, ax2 = lut.axes[1], lut.axes[0]
        name1, name2 = lut.names[1], lut.names[0]
        data = np.swapaxes(lut.data, 0, 1)
    else:
        ax1, ax2 = lut.axes[0], lut.axes[1]
        name1, name2 = lut.names[0], lut.names[1]
        data = lut.data


    if vmin is None:
        vmin = np.amin(lut.data[~np.isnan(lut.data)])
    if vmax is None:
        vmax = np.amax(lut.data[~np.isnan(lut.data)])
    if vmin == vmax:
        vmin -= 0.001
        vmax += 0.001
    if vmin > vmax: vmin, vmax = vmax, vmin
    if percent:
        vmin=0.
        vmax=100.

    #
    # semi polar axis
    #
    # angle
    ax1_scaled = ax1
    label2 = name2

    # convert Idx instance to index if necessarry
    if isinstance(index, Idx_base):
        index = int(np.around(index.index(ax1)))
    mirror_index = (ax1_scaled.shape[0]//2 + index)%ax1_scaled.shape[0]

    if swap:
        title=lut.axes[1][index]
    else:
        title=lut.axes[0][index]

    # radius axis
    # scale to [0, 90]
    ax2_min = np.amin(ax2)
    ax2_max = np.amax(ax2)
    label1 = name1 + ' {:7.2f}'.format(title)
    #

    ax_cart = fig.add_subplot(sub)
    ax_cart.grid(True)

    if sym:
        ax_cart.set_xlim(-ax2_max, ax2_max)
    else:
        ax_cart.set_xlim(ax2_min, ax2_max)
    ax_cart.set_ylim(vmin, vmax)
    ax_cart.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax_cart.grid(True)
    ax_cart.set_xlabel(label2)
    #ax_cart.set_title(label1)

    #
    # plot transects
    #
    ax_cart.plot(ax2, data[index,:],fmt, color=color)
    if sym:
       ax_cart.plot(-ax2, data[mirror_index,:],fmt, color=color)

    if lut.desc is not None:
        ax_cart.set_title(lut.desc)


def merge(M, axes, dtype=None):
    '''
    Merge several luts

    Arguments:
        - M is a list of MLUT objects to merge
        - axes is a list of axes names to merge
          these names should be present in each LUT attribute
        - dtype is the data type of the new axes
          ex: dtype=float
          if None, no data type conversion

    Returns a MLUT for which each dataset has new axes as defined in list axes
    (list of strings)
    The attributes of the merged mlut consists of the common attributes with
    identical values.

    >>> np.random.seed(0)

    Example: merge two MLUTS
    (also using attribute to dataset promotion)
    >>> M = []
    >>> for b in range(4):
    ...     M1 = MLUT()
    ...     M1.add_axis('ax1', np.arange(4))
    ...     M1.add_axis('ax2', np.arange(5)+10)
    ...     M1.add_dataset('a', np.random.randn(4, 5), ['ax1', 'ax2'])
    ...     M1.set_attrs({'b':b, 'c':b*10})
    ...     M1.promote_attr('b')  # attribute 'b' is converted to a scalar dataset
    ...     M.append(M1)
    >>> merged = merge(M, ['c'])
    >>> _=merged.print_info(show_self=False)
     Datasets:
      [0] a (float64 between -2.55 and 2.27), axes=('c', 'ax1', 'ax2')
      [1] b (float64 between 0 and 3), axes=('c',)
     Axes:
      [0] ax1: 4 values between 0 and 3
      [1] ax2: 5 values between 10 and 14
      [2] c: 4 values between 0 and 30

    '''

    m = MLUT()
    first = M[0]

    # check mluts compatibility
    for i in xrange(1, len(M)):
        assert first.equal(M[i], content=False, attributes=False, show_diff=True)

    # add old axes
    for (axname, axis) in first.axes.items():
        m.add_axis(axname, axis)

    # determine the new axes from the attributes of all mluts
    newaxes = []  # new axes
    newaxnames = []
    for axname in axes:
        axis = []
        for mlut in M:
            value = mlut.attrs[axname]
            if dtype is not None:
                value = dtype(value)
            if value not in axis:
                axis.append(value)
        m.add_axis(axname, axis)
        newaxes.append(axis)
        newaxnames.append(axname)

    # dataset loop
    for name in first.datasets():

        # build new data
        new_shape = tuple(map(len, newaxes))+first[name].shape
        _dtype = first[name].data.dtype
        newdata = np.zeros(new_shape, dtype=_dtype)
        try:
            newdata += np.NaN
        except:
            pass
        for mlut in M:

            # find the index of the attributes in the new LUT
            index = ()
            for j, a in enumerate(axes):
                value = mlut.attrs[a]
                if dtype is not None:
                    value = dtype(value)
                index += (newaxes[j].index(value),)

            if first[name].data.ndim != 0:
                index += (slice(None),)

            newdata[index] = mlut[name].data

        axnames = first[name].names
        if axnames is None:
            m.add_dataset(name, newdata)
        else:
            m.add_dataset(name, newdata, newaxnames+axnames)

    # fill with common arguments
    for k, v in first.attrs.items():
        if False in map(lambda x: k in x.attrs, M):
            continue
        if isinstance(v, np.ndarray):
            if False in map(lambda x: np.allclose(v, x.attrs[k]), M):
                continue
        else:
            if False in map(lambda x: v == x.attrs[k], M):
                continue
        m.set_attr(k, v)

    return m


class MLUT(object):
    '''
    A class to store and manage multiple look-up tables

    How to create a MLUT:
    >>> m = MLUT()
    >>> m.add_axis('a', np.linspace(100, 150, 5))
    >>> m.add_axis('b', np.linspace(5, 8, 6))
    >>> m.add_axis('c', np.linspace(0, 1, 7))
    >>> np.random.seed(0)
    >>> m.add_dataset('data1', np.random.randn(5, 6), ['a', 'b'])
    >>> m.add_dataset('data2', np.random.randn(5, 6, 7), ['a', 'b', 'c'])
    >>> # Add a dataset without associated axes
    >>> m.add_dataset('data3', np.random.randn(10, 12))
    >>> m.set_attr('x', 12)   # set MLUT attributes
    >>> m.set_attrs({'y':15, 'z':8})
    >>> _=m.print_info(show_self=False)
     Datasets:
      [0] data1 (float64 between -2.55 and 2.27), axes=('a', 'b')
      [1] data2 (float64 between -2.22 and 2.38), axes=('a', 'b', 'c')
      [2] data3 (float64 between -2.77 and 2.3), axes=(None, None)
     Axes:
      [0] a: 5 values between 100.0 and 150.0
      [1] b: 6 values between 5.0 and 8.0
      [2] c: 7 values between 0.0 and 1.0

    Use bracket notation to extract a LUT
    Note that you can use a string or integer.
    data1 is the first dataset in this case, we could use m[0]
    >>> _=m['data1'].print_info()  # or m[0]
    LUT "data1" (float64 between -2.55 and 2.27):
      Dim 0 (a): 5 values betweeen 100.0 and 150.0
      Dim 1 (b): 6 values betweeen 5.0 and 8.0
    '''
    def __init__(self):
        # axes
        self.axes = OrderedDict()
        # data: a list of (name, array, axnames, attributes)
        self.data = []
        # attributes
        self.attrs = OrderedDict()

    def datasets(self):
        ''' returns a list of the datasets names '''
        return [x[0] for x in self.data]

    def add_axis(self, name, axis):
        ''' Add an axis to the MLUT '''
        assert isinstance(name, str)
        assert name not in self.axes, 'Axis "{}" already in MLUT'.format(name)
        if isinstance(axis, list):
            ax = np.array(axis)
        else:
            ax = axis
        assert ax.ndim == 1

        self.axes[name] = ax

    def add_dataset(self, name, dataset, axnames=None, attrs={}):
        '''
        Add a dataset to the MLUT
        name (str): name of the dataset
        dataset (np.array)
        axnames: list of (strings or None), or None
        attrs: dataset attributes
        '''
        assert name not in [x[0] for x in self.data], 'Error, "{}" already in MLUT'.format(name)
        if axnames is not None:
            # check axes consistency
            assert len(axnames) == len(dataset.shape)
            for i, ax in enumerate(axnames):
                if ax is None: continue
                if ax not in self.axes: continue
                assert dataset.shape[i] == len(self.axes[ax])
        else:
            axnames = [None]*dataset.ndim

        self.data.append((name, dataset, axnames, attrs))

    def add_lut(self, lut, desc=None):
        '''
        Add a LUT to the MLUT

        returns self
        '''
        if desc is None:
            desc = lut.desc
        assert desc is not None
        for iax in xrange(lut.ndim):
            axname = lut.names[iax]
            ax = lut.axes[iax]
            if axname in self.axes:
                # check axis
                if ax is not None:
                    assert np.array(self.axes[axname]).shape == np.array(ax).shape, \
                            'Inconsistent shapes for axis "{}": {} != {}'.format(
                                    axname, self.axes[axname].shape, ax.shape)
                    assert np.allclose(self.axes[axname], ax)
            elif axname is None:
                assert ax is None
            elif ax is not None:
                # add the axis
                self.add_axis(axname, ax)

        self.add_dataset(desc, lut.data, axnames=lut.names, attrs=lut.attrs)

        return self

    def rm_lut(self, name):
        ''' remove a LUT '''
        try:
            assert isinstance(name, basestring)
        except NameError: # must be python 3
            assert isinstance(name, (str, bytes))

        try:
            index = [x[0] for x in self.data].index(name)
        except ValueError:
            raise Exception('{} is not in {}', name, self)

        self.data.pop(index)

    def sub(self, d):
        '''
        The MLUT equivalent of LUT.sub

        returns a MLUT where each LUT is subsetted using dictionary d
        keys should only be strings
        values can be int, float, slice, Idx, array (bool or int)
        '''
        m = MLUT()
        for dd in d:
            assert isinstance(dd, str)

        for dd in self.datasets():
            m.add_lut(self[dd].sub(d, ignore=True))

        m.attrs = self.attrs

        return m

    def save(self, filename, fmt=None, overwrite=False,
             verbose=False, compress=True):
        '''
        Save a MLUT to filename
        fmt: output format: hdf4, netcdf4,
             or None (determine from filename extension)
        '''

        if exists(filename):
            if overwrite:
                remove(filename)
            else:
                ex = Exception('File {} exists'.format(filename))
                setattr(ex, 'filename', filename)
                raise ex

        if verbose:
            print('Writing "{}" to "{}" ({} format)'.format(self, filename, fmt))

        if fmt is None:
            if filename.endswith('.hdf'):
                fmt = 'hdf4'
            elif filename.endswith('.nc'):
                fmt = 'netcdf4'
            else:
                raise ValueError('Cannot determine desired format '
                        'of filename "{}"'.format(filename))

        if fmt=='netcdf4':
            self.__save_netcdf4(filename, overwrite=overwrite,
                              verbose=verbose, compress=compress)
        elif fmt=='hdf4':
            self.__save_hdf(filename, overwrite=overwrite,
                          verbose=verbose, compress=compress)
        else:
            raise ValueError('Invalid format {}'.format(fmt))

    def __save_netcdf4(self, filename, overwrite=False,
                      verbose=False, compress=True):
        from netCDF4 import Dataset
        root = Dataset(filename, 'w', format='NETCDF4')
        dummycount = 0

        # write axes and create associated dimensions
        for axname, ax in self.axes.items():
            root.createDimension(axname, len(ax))
            var = root.createVariable(axname, ax.dtype, [axname], zlib=compress)
            var[:] = ax[:]

        # write datasets
        for (name, data, axnames, attributes) in self.data:
            if verbose:
                print('   Write data "{}" ({}, {})'.format(name, data.dtype, data.shape))
            # create dummy dimensions when axis is missing
            for i in xrange(len(axnames)):
                if axnames[i] is None:
                    dummycount += 1
                    axnames[i] = 'dummy{:d}'.format(dummycount)
                if axnames[i] not in root.dimensions:
                    if verbose:
                        print('   Create dimension {}'.format(axnames[i]))
                    root.createDimension(axnames[i], data.shape[i])

            var = root.createVariable(name, data.dtype, axnames, zlib=compress)
            var.setncatts(attributes)
            var[:] = data[...]

        # write global attributes
        if verbose:
            print('   Write {} attributes'.format(len(self.attrs)))
        root.setncatts(self.attrs)

        root.close()


    def __save_hdf(self, filename, overwrite=False, verbose=False, compress=True):
        '''
        Save a MLUT to a hdf file
        '''
        from pyhdf.SD import SD, SDC
        from pyhdf.error import HDF4Error

        def safecast(data):
            # hdf4 does not support int64, uint64
            # cast to 32-bits
            if data.dtype == np.dtype('uint64'):
                assert np.allclose(data, data.astype('uint32'))
                return data.astype('uint32')
            if data.dtype == np.dtype('int64'):
                assert np.allclose(data, data.astype('int32'))
                return data.astype('int32')
            return data


        typeconv = {
                    np.dtype('float32'): SDC.FLOAT32,
                    np.dtype('float64'): SDC.FLOAT64,
                    np.dtype('uint64'): SDC.UINT32, # /!\
                    np.dtype('int64'): SDC.INT32,   # hdf4 does not support 64-bit ints
                    np.dtype('uint32'): SDC.UINT32,
                    np.dtype('int32'): SDC.INT32,
                    np.dtype('uint16'): SDC.UINT16,
                    np.dtype('int16'): SDC.INT16,
                    np.dtype('uint8'): SDC.UINT8,
                    np.dtype('int8'): SDC.INT8,
                    }
        hdf = SD(filename, SDC.WRITE | SDC.CREATE)

        # write axes
        if self.axes is not None:
            for name, ax in self.axes.items():
                if verbose:
                    print('   Write axis "{}" in "{}"'.format(name, filename))
                type = typeconv[ax.dtype]
                sds = hdf.create(name, type, ax.shape)
                if compress:
                    sds.setcompress(SDC.COMP_DEFLATE, 9)
                sds[:] = safecast(ax)[:]
                sds.endaccess()

        # write datasets
        for name, data, axnames, attrs in self.data:

            data = safecast(data)

            if verbose:
                print('   Write data "{}" ({}, {})'.format(name, data.dtype, data.shape))
            type = typeconv[data.dtype]

            # write data
            if data.ndim == 0:
                # scalar
                sds = hdf.create(name, type, (1,))
                setattr(sds, 'lut:scalar', 'True')
            else:
                sds = hdf.create(name, type, data.shape)
            if compress:
                sds.setcompress(SDC.COMP_DEFLATE, 9)
            sds[:] = data[...]

            if axnames not in [None, []]:
                setattr(sds, 'dimensions', ','.join(map(str, axnames)))
            if 'dimensions' in attrs:
                raise Exception('Error writing {}, "dimensions" attribute conflict'.format(filename))
            for k, v in attrs.items():
                setattr(sds, k, v)
            sds.endaccess()

        # write attributes
        if verbose:
            print('   Write {} attributes'.format(len(self.attrs)))
        for k, v in self.attrs.items():
            try:
                setattr(hdf, k, v)
            except HDF4Error:
                setattr(hdf, str(k), str(v))

        hdf.end()

    def set_attr(self, key, value):
        '''
        Set one attribute key -> value
        '''
        self.attrs[key] = value

        return self

    def set_attrs(self, attributes):
        '''
        Set multiple attributes to attrs
        attributes: dict
        '''
        self.attrs.update(attributes)

        return self

    def print_info(self, *args, **kwargs):
        # same as describe()
        return self.describe(*args, **kwargs)

    def describe(self, show_range=True, show_self=True, show_attrs=False, show_shape=False, show_axes=True, mem=False):
        total_size = 0
        if show_self:
            print(str(self))
        print(' Datasets:')
        for i, (name, dataset, axes, attrs) in enumerate(self.data):
            axdesc = ''
            if (axes is not None) and show_axes:
                axdesc += ', axes='+ str(tuple(axes))
            if show_shape:
                axdesc += ', shape={}'.format(dataset.shape)
            if show_range and isinstance(dataset, np.ndarray):
                try:
                    rng = ' in [{:.3g}, {:.3g}]'.format(np.amin(dataset), np.amax(dataset))
                except:
                    rng = ''
            else:
                rng = ''
            if mem:
                memdesc = ', {}'.format(sizeof_fmt(dataset.nbytes))
            else:
                memdesc = ''
            print('  [{}] {} ({}{})'.format(i, name, dataset.dtype, rng, dataset.shape) + axdesc + memdesc)
            total_size += dataset.nbytes

            if show_attrs and (len(attrs) != 0):
                print('    Attributes:')
                for k, v in attrs.items():
                    print('      {}: {}'.format(k, v))
        print(' Axes:')
        for i, (name, values) in enumerate(self.axes.items()):
            print('  [{}] {}: {} values in [{}, {}]'.format(i, name, len(values), values[0], values[-1]))
        if show_attrs:
            print(' Attributes:')
            for k, v in self.attrs.items():
                print(' ', k, ':', v)
        if mem:
            print('Total memory usage: {}'.format(sizeof_fmt(total_size)))

        return self

    def dropaxis(self, *ax_to_drop):
        '''
        drop axes of size 1

        example:
            m.dropaxis('a')   # remove axis 'a' of size 1
            m.dropaxis('a', 'b')   # remove 'a' and 'b'
        '''
        m = MLUT()

        # axes
        for axname, ax in self.axes.items():
            if axname not in ax_to_drop:
                m.add_axis(axname, ax)

        # datasets
        for (name, data, axnames, attributes) in self.data:
            axes = [x for x in axnames if x not in ax_to_drop]
            shp = [s for (x, s) in zip(axnames, data.shape) if x not in ax_to_drop]
            m.add_dataset(name, data.reshape(shp), axnames=axes, attrs=attributes)

        # global attributes
        m.attrs = self.attrs

        return m


    def promote_attr(self, name):
        '''
        Create a new dataset from attribute name
        '''
        assert isinstance(name, str)
        assert name in self.attrs
        value = np.array(self.attrs[name])

        self.add_dataset(name, value)

    def __getitem__(self, key):
        '''
        return the LUT corresponding to key (int or string)
        '''
        if isinstance(key, (str, unicode)):
            index = -1
            for i, (name, _, _, _) in enumerate(self.data):
                if key == name:
                    index = i
                    break
            if index == -1:
                raise Exception('Cannot find dataset {}'.format(key))
        elif isinstance(key, int):
            index = key
        else:
            raise Exception('multi-dimensional LUTs should only be indexed with strings or integers')

        name, dataset, axnames, attrs = self.data[index]
        if axnames is None:
            axes = None
        else:
            axes = []
            for ax in axnames:
                if (ax is None) or (ax not in self.axes):
                    axes.append(None)
                else:
                    axes.append(self.axes[ax])

        return LUT(desc=name, data=dataset, axes=axes, names=axnames, attrs=attrs)

    def equal(self, other, content=True, attributes=True, show_diff=False):
        '''
        Test equality between two MLUTs
        Arguments:
         * show_diff: print their differences
         * content: check LUT content (otherwise only axes and shapes)
         * attributes: check global attributes
        '''
        msg = 'MLUTs diff:'
        if not isinstance(other, MLUT):
            msg += '  other is not a MLUT ({})'.format(str(other))
            print(msg)
            return False

        eq = True

        # check axes
        for k in set(self.axes).union(other.axes):
            if (k not in other.axes) or (k not in self.axes):
                msg += '  axis {} missing in either\n'.format(k)
                eq = False
            if self.axes[k].shape != other.axes[k].shape:
                msg += '  axis {} shape mismatch\n'.format(k)
                eq = False
            if not np.allclose(self.axes[k], other.axes[k]):
                msg += '  axis {} is different\n'.format(k)
                eq = False

        # check datasets
        if set(self.datasets()) != set(other.datasets()):
            msg += '  Datasets are different\n'
            msg += '   -> {}'.format(str(self.datasets()))
            msg += '   -> {}'.format(str(other.datasets()))
            eq = False
        for name in self.datasets():
            if not self[name].equal(other[name], strict=content):
                msg += '  dataset {} differs (shapes are {} and {})\n'.format(name, self[name].shape, other[name].shape)
                eq = False

        # check global attributes
        if attributes:
            for a in set(self.attrs.keys()).union(other.attrs.keys()):
                if (a not in self.attrs) or (a not in other.attrs):
                    msg += '  attribute {} missing in either MLUT\n'.format(a)
                    eq = False
                    continue
                if (self.attrs[a] != other.attrs[a]):
                    msg += '  value of attribute {} differs ({} and {})\n'.format(a, self.attrs[a], other.attrs[a])
                    eq = False
                    continue

        if show_diff and not eq:
            print(msg)

        return eq

    def __eq__(self, other):
        return self.equal(other)

    def __neq__(self, other):
        return not self.equal(other)

    def axis(self, axname, aslut=False):
        '''
        returns an axis
        if aslut: returns it as a LUT
        otherwise, as values
        '''
        data = self.axes[axname]
        if aslut:
            return LUT(desc=axname, data=data, axes=[data], names=[axname])
        else:
            return data

    def plot(self, datasets=None, extra_widgets=True, *args, **kwargs):
        '''
        display all datasets in the MLUT
        * datasets: list of datasets to display
                    if None, display all datasets
        * extra_widgets: show extra widgets for:
                 1) interactive selection of datasets to display
                 2) choice of min/max values
        '''
        try:
            from ipywidgets import VBox, HBox, Checkbox, IntSlider, HTML, FloatText, Button
            from pylab import axis
            from IPython.display import display, clear_output
        except ImportError:
            raise Exception('IPython notebook widgets are required '
                    'to plot a MLUT')

        if datasets is None:
            datasets = self.datasets()

        wid = []
        axes = {}  # name: ()
        dstchk = {}  # dataset checkboxes
        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
        else:
            vmin = 0
        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
        else:
            vmax = 1
        vminmax = [   # vmin/vmax/xmin/xmax float texts
                FloatText(description='vmin', value=vmin),
                FloatText(description='vmax', value=vmax),
                Checkbox(description='xmin/xmax', value=False),
                FloatText(description='xmin', value=0),
                FloatText(description='xmax', value=1),
                ]
        for d in datasets:
            dstchk[d] = Checkbox(description=d, value=False)
            for iax, name in enumerate(self[d].names):
                if name is None:
                    raise Exception('Does not work with unnamed axes, sorry :/')
                if name in axes:
                    continue

                ax = self[d].axes[iax]
                desc = name
                if ax is not None:
                    desc += ' [{:.5g}, {:.5g}]'.format(ax[0], ax[-1])

                chk = Checkbox(description=desc, value=True)
                slider = IntSlider(min=0, max=self[d].shape[iax]-1)
                slider.visible = False
                text = HTML(value='')
                text.visible = False
                button_minus = Button(description='-')
                setattr(button_minus, 'slider', slider)
                button_plus  = Button(description='+')
                setattr(button_plus, 'slider', slider)

                wid.append(HBox([chk, text, slider, button_minus, button_plus]))

                axes[name] = (chk, slider, text, button_minus, button_plus)

        if extra_widgets:
            wid.insert(0, HTML(value='<b>AXES:</b>'))
            wid.insert(0, HBox(dstchk.values()))
            wid.insert(0, HTML(value='<b>DATASETS:</b>'))
            wid.insert(0, HBox(vminmax))

        def update():
            clear_output()

            # update vmin/vmax
            if extra_widgets:
                kwargs['vmin'] = vminmax[0].value
                kwargs['vmax'] = vminmax[1].value
                vminmax[3].visible = vminmax[2].value
                vminmax[4].visible = vminmax[2].value

            # update sliders visibility
            for a, (chk, slider, text, button_minus, button_plus) in axes.items():
                slider.visible = chk.value
                text.visible = chk.value
                button_minus.visible = chk.value
                button_plus.visible = chk.value

            # display each dataset
            for d in datasets:

                if not dstchk[d].value:
                    continue

                keys = []
                ndim = 0

                for i, name in enumerate(self[d].names):
                    chk, slider, text, _, _ = axes[name]

                    if chk.value:
                        index = slider.value
                        keys.append(index)
                        # set text (value)
                        if self[d].axes[i] != None:
                            text.value = '&nbsp;&nbsp;{:.5g}&nbsp;&nbsp;'.format(self[d].axes[i][index])
                    else:
                        keys.append(slice(None))
                        ndim += 1

                if ndim <= 2:
                    self[d].sub().__getitem__(tuple(keys)).plot(label=d, legend=True, *args, **kwargs)
                    if vminmax[2].value:
                        axis(xmin=vminmax[3].value, xmax=vminmax[4].value)
                else:
                    print('{}: Please select at least {} dimension(s)'.format(d, ndim-2))
        def decrement(b):
            b.slider.value -= 1
        def increment(b):
            b.slider.value += 1
        for a, (chk, slider, text, button_minus, button_plus) in axes.items():
            chk.on_trait_change(update, 'value')
            slider.on_trait_change(update, 'value')
            button_minus.on_click(decrement)
            button_plus.on_click(increment)
        for fl in vminmax:
            fl.on_trait_change(update, 'value')
        for chk in dstchk.values():
            chk.on_trait_change(update, 'value')

        display(VBox(wid))
        update()



def read_mlut(filename, fmt=None):
    '''
    Read a MLUT (multi-format)
    fmt: netcdf4, hdf4
         or None (determine format from extension)
    '''
    if fmt is None:
        if filename.endswith('.hdf'):
            fmt = 'hdf4'
        elif filename.endswith('.nc'):
            fmt = 'netcdf4'
        else:
            raise ValueError('Cannot determine desired format '
                    'of filename "{}"'.format(filename))

    if fmt=='netcdf4':
        return read_mlut_netcdf4(filename)
    elif fmt=='hdf4':
        return read_mlut_hdf(filename)
    elif fmt=='hdf5':
        return read_mlut_hdf5(filename)

    else:
        raise ValueError('Invalid format {}'.format(fmt))


def read_mlut_netcdf4(filename):
    '''
    Read a MLUT (netcdf4 format)
    '''
    # assumes everything is in the root group
    m = MLUT()
    from netCDF4 import Dataset
    root = Dataset(filename, 'r', format='NETCDF4')

    # read axes
    for dim in root.dimensions:
        if (not dim.startswith('dummy')) and (dim in root.variables):
            m.add_axis(str(dim), root.variables[dim][:])

    # read datasets
    for varname in root.variables:
        if varname in m.axes:
            continue
        var = root.variables[varname]

        # read attributes
        attrs = {}
        for a in var.ncattrs():
            attrs[a] = var.getncattr(a)

        m.add_dataset(varname, var[:], [str(x) for x in var.dimensions], attrs=attrs)

    # read global attributes
    for a in root.ncattrs():
        m.set_attr(a, root.getncattr(a))

    root.close()

    return m


def read_mlut_hdf5(filename, datasets=None, lazy=False, group=None):
    '''
    read a MLUT from a hdf5 file (filename)
    datasets: list of datasets to read:
        * None (default): read all datasets, including axes as indicated by the
          attribute 'dimensions'
        * a list of:
            - dataset names (string)
            - or a tuple (dataset_name, axes) where axes is a list of
              dimensions (strings), overriding the attribute 'dimensions'
    '''
    import h5py

    ff = h5py.File(filename)

    if group:
        f = ff[group]
    else:
        f = ff

    # set the list of dataset
    if datasets is None:
        ls_datasets = f['data'].keys()
    else:
        ls_datasets = datasets

    # look for axis rquired for the datasets
    ls_axis   = []
    axis_data = []
    for dataset in ls_datasets: 
        if dataset in f['data'].keys():
            if not f['data'][dataset].attrs.__contains__('dimensions'):
                print('Missing -dimensions- Attr in dataset "{}" '.format(dataset))
                raise Exception('Missing dimensions Attr in dataset')
            else:
                dimensions = f['data'][dataset].attrs.get('dimensions').split(',')
                axis_data.append(dimensions)
                for aa in dimensions:
                    ls_axis.append(aa)
        else:
            print('dataset "{}" not available.'.format(dataset))
            print('{} contains the following datasets:'.format(filename))
            for d in f['data'].keys():
                print ('  *', d)
            raise Exception('Missing dataset')
    ls_axis = list(set(ls_axis))

    m = MLUT()
    # add axis to the MLUT
    for ax in ls_axis:
        axis = f['axis'][ax][...]
        m.add_axis(ax, axis)

    # add data to MLUT
    for idata in xrange(len(ls_datasets)):
        dataset = ls_datasets[idata]
        if lazy:
            data = f['data'][dataset]
        else:
            data = f['data'][dataset][...]
        attrs = {}
        if f['data'][dataset].attrs.__contains__('_FillValue'):
            attrs['_FillValue'] = f['data'][dataset].attrs.get('_FillValue')
        if f['data'][dataset].attrs.__contains__('add_offset'):
            attrs['add_offset'] = f['data'][dataset].attrs.get('add_offset')
        if f['data'][dataset].attrs.__contains__('scale_factor'):
            attrs['scale_factor'] = f['data'][dataset].attrs.get('scale_factor')
        m.add_dataset(dataset, data, axnames=axis_data[idata], attrs=attrs)

    if not lazy:
        ff.close()

    return m


def read_mlut_hdf(filename, datasets=None):
    '''
    read a MLUT from a hdf file (filename)
    datasets: list of datasets to read:
        * None (default): read all datasets, including axes as indicated by the
          attribute 'dimensions'
        * a list of:
            - dataset names (string)
            - or a tuple (dataset_name, axes) where axes is a list of
              dimensions (strings), overriding the attribute 'dimensions'
    '''
    from pyhdf.SD import SD

    hdf = SD(filename)

    # read the datasets
    ls_axes = []
    ls_datasets = []
    if datasets is None:
        datasets = xrange(len(hdf.datasets()))
    else:
        assert isinstance(datasets, list), 'datasets should be provided as a list'

    for i in datasets:
        if isinstance(i, tuple):
            (name, axes) = i
            sds = hdf.select(name)
        else:
            axes = None
            sds = hdf.select(i)
        sdsname = sds.info()[0]

        if (axes is None) and ('dimensions' in sds.attributes()):
            axes = sds.attributes()['dimensions'].split(',')
            axes = [x.strip() for x in axes]

            # replace 'None's by None
            axes = [None if (x=='None') else x for x in axes]

        if axes is not None:
            ls_axes.extend(axes)

        data = sds.get()
        attrs = sds.attributes()
        if 'lut:scalar' in attrs:
            attrs.pop('lut:scalar')
            data = data.reshape(())
        ls_datasets.append((sdsname, data, axes, attrs))

    # remove 'None' axes
    while None in ls_axes:
        ls_axes.remove(None)

    # transfer the axes from ls_datasets to the new MLUT
    m = MLUT()
    for ax in set(ls_axes):
        [x[0] for x in ls_datasets]

        # read the axis if not done already
        if ax not in [x[0] for x in ls_datasets]:
            if ax in hdf.datasets():
                sds = hdf.select(ax)
                m.add_axis(ax, sds.get())
        else:
            i = [x[0] for x in ls_datasets].index(ax)
            (name, data, _, _) = ls_datasets.pop(i)
            m.add_axis(name, data)

    # add the datasets
    for (name, data, axnames, attrs) in ls_datasets:
        if 'dimensions' in attrs:
            attrs.pop('dimensions')
        m.add_dataset(name, data, axnames, attrs)

    # read the global attributes
    for k, v in hdf.attributes().items():
        m.set_attr(k, v)

    return m



if __name__ == '__main__':
    import doctest
    doctest.testmod()
