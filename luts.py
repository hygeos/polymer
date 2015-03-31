#!/usr/bin/env python
# encoding: utf-8

'''
Several tools for look-up tables management and interpolation

Provides:
    - LUT class: extends ndarrays for generic multi-dimensional interpolation
    - Idx class: find the index of values, for LUT interpolation
    - merge: look-up tables merging
    - read_lut_hdf: read LUTs from HDF files
'''

import numpy as np
from pyhdf.SD import SD, SDC
from scipy.interpolate import interp1d
from os.path import exists
from os import remove


class LUT(object):
    '''
    Look-up table storage with generic multi-dimensional interpolation.
    Extends the __getitem__ method of ndarrays to float and float arrays (index
    tables with floats)
    The LUT axes can be optionally provided so that values can be interpolated
    into float indices in a first step, using the Idx class.

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
            self.attrs = {}
        else:
            self.attrs = attrs
        self.ndim = self.data.ndim
        self.shape = data.shape
        self.sub = Subsetter(self)

        # check axes
        if axes is None:
            self.axes = self.ndim * [None]
        else:
            self.axes = axes
            assert len(axes) == self.ndim
            for ax in axes:
                if isinstance(ax, np.ndarray):
                    assert ax.ndim == 1
                elif isinstance(ax, list): pass
                elif ax is None: pass
                else:
                    raise Exception('Invalid axis type {}'.format(ax.__class__))

        # check names
        if names is None:
            self.names = self.ndim * [None]
        else:
            self.names = names
            assert len(names) == self.ndim

    def print_info(self, prepend=''):
        if self.desc is None:
            print prepend+'LUT dimensions ({}):'.format(self.data.dtype)
        else:
            print prepend+'LUT dimensions ({}, {}):'.format(self.desc, self.data.dtype)

        for i in xrange(self.data.ndim):
            if self.names[i] is None:
                name = 'NoName'
            else:
                name = self.names[i]
            if self.axes[i] is None:
                print prepend+'  Dim {} ({}): No axis attached'.format(i, name)
            else:
                print prepend+'  Dim {} ({}): {} values betweeen {} and {}'.format(
                        i, name,
                        len(self.axes[i]),
                        self.axes[i][0],
                        self.axes[i][-1])

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
            - Unlike ndarrays, the number of dimensions in keys should be
              identical to the dimension of the LUT
              >>> LUT(np.zeros((2, 2)))[:]
              Traceback (most recent call last):
              ...
              Exception: Incorrect number of dimensions in __getitem__

        Returns: a scalar or ndarray
        '''

        if not isinstance(keys, tuple):
            keys = (keys,)
        keys = list(keys)
        N = len(keys)

        if N != self.ndim:
            raise Exception('Incorrect number of dimensions in __getitem__')

        # determine the interpolation axes
        # and for those axes, determine the lower index (inf) and the weight
        # (x) between lower and upper index
        interpolate_axis = []   # indices of the interpolated axes
        inf_list = []       # index of the lower elements for the interpolated axes
        x_list = []         # weights, for the interpolated axes
        for i in xrange(N):
            k = keys[i]

            # convert Idx keys
            if isinstance(k, Idx):
                # k is an Idx instance
                # convert it to float indices for the current axis
                if k.name not in [None, self.names[i]]:
                    msg = 'Error, wrong parameter passed at position {}, expected {}, got {}'
                    raise Exception(msg.format(i, self.names[i], k.name))
                k = k.index(self.axes[i])
                keys[i] = k

            # floating-point indices should be interpolated
            interpolate = False
            if isinstance(k, np.ndarray) and (k.dtype in [np.dtype('float')]):
                interpolate = True
                inf = k.astype('int')
                inf[inf == self.data.shape[i]-1] -= 1
            elif isinstance(k, float):
                interpolate = True
                inf = int(k)
                if inf == self.data.shape[i]-1:
                    inf -= 1
            if interpolate:
                # current axis needs interpolation
                inf_list.append(inf)
                x_list.append(k-inf)
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

    def check_compatible(self, other):
        '''
        if other is a LUT, check that it is 'compatible' to self (ie, they have
        same axes and metadata), and return its data
        otherwise return other as-is
        '''

        if isinstance(other, LUT):

            # check that axes are all equal
            assert np.all(map(lambda x, y: np.all([x==y]), self.axes, other.axes))

            # check that names are all equal
            assert self.names == other.names

            # same for desc...
            if self.desc == other.desc:
                desc = self.desc
            else:
                desc = None

            return other.data, desc

        else:
            return other, self.desc

    def __add__(self, other):
        '''
        sum of two LUTs
        '''
        otherdata, desc = self.check_compatible(other)

        return LUT(self.data + otherdata,
                axes=self.axes,
                names=self.names,
                desc=desc,
                attrs=self.attrs)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        '''
        difference between two LUTs
        '''
        otherdata, desc = self.check_compatible(other)

        return LUT(self.data - otherdata,
                axes=self.axes,
                names=self.names,
                desc=desc,
                attrs=self.attrs)

    def __rsub__(self, other):
        '''
        difference between two LUTs
        '''
        otherdata, desc = self.check_compatible(other)

        return LUT(otherdata - self.data,
                axes=self.axes,
                names=self.names,
                desc=desc,
                attrs=self.attrs)

    def __mul__(self, other):
        '''
        multiply a LUT
        '''
        otherdata, desc = self.check_compatible(other)

        return LUT(self.data * otherdata,
                axes=self.axes,
                names=self.names,
                desc=desc,
                attrs=self.attrs)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        '''
        divide a LUT
        '''
        otherdata, desc = self.check_compatible(other)

        return LUT(self.data / otherdata,
                axes=self.axes,
                names=self.names,
                desc=desc,
                attrs=self.attrs)

    def __rdiv__(self, other):
        otherdata, desc = self.check_compatible(other)

        return LUT(otherdata / self.data,
                axes=self.axes,
                names=self.names,
                desc=desc,
                attrs=self.attrs)

    def save(self, filename):
        '''
        save a LUT in a HDF file
        if the file exists already, add the LUT to it (the axes should be
        compatible)
        '''
        assert self.desc is not None
        assert self.names is not None

        print 'Writing "{}" to "{}"'.format(self.desc, filename)
        hdf = SD(filename, SDC.WRITE | SDC.CREATE)

        # check that data has not been written yet
        if self.desc in hdf.datasets():
            raise Exception('Cannot write {}: it is already in {}'.format(self.desc, filename))

        # check axes
        for i in xrange(self.ndim):
            name = self.names[i]
            ax = self.axes[i]
            if name is None:
                raise Exception('LUT.save() requires named axes')
            if name in hdf.datasets():
                sds = hdf.select(name)
                rank = sds.info()[1]
                shape = sds.info()[2]
                assert rank == 1, 'Axis {} does not have a rank of 1'.format(name)
                assert ax.shape[0] == shape

        # write the axes if necessary
        for i in xrange(self.ndim):
            name = self.names[i]
            ax = np.array(self.axes[i])
            if name not in hdf.datasets():
                print '   Write axis "{}" in "{}"'.format(name, filename)
                type = {
                        np.dtype('float32'): SDC.FLOAT32,
                        np.dtype('float64'): SDC.FLOAT64,
                        }[ax.dtype]
                sds = hdf.create(name, type, ax.shape)
                sds.setcompress(SDC.COMP_DEFLATE, 9)
                sds[:] = ax[:]
                sds.endaccess()

        # write data
        print '   Write data "{}"'.format(self.desc)
        type = {
                np.dtype('float32'): SDC.FLOAT32,
                np.dtype('float64'): SDC.FLOAT64,
                }[self.data.dtype]
        sds = hdf.create(self.desc, type, self.data.shape)
        sds.setcompress(SDC.COMP_DEFLATE, 9)
        sds[:] = self.data[:]
        setattr(sds, 'dimensions', ','.join(self.names))
        sds.endaccess()

        hdf.end()


class Subsetter(object):
    '''
    A conveniency class to use the syntax like:
    LUT.sub[:,:,0]
    for subsetting LUTs
    '''
    def __init__(self, LUT):
        self.LUT = LUT

    def __getitem__(self, keys):
        '''
        subset parent LUT
        '''
        axes = []
        names = []
        attrs = self.LUT.attrs
        desc = self.LUT.desc

        for i in xrange(self.LUT.ndim):
            if keys[i] == slice(None):
                axes.append(self.LUT.axes[i])
                names.append(self.LUT.names[i])

        data = self.LUT.__getitem__(keys)

        return LUT(data, axes=axes, names=names, attrs=attrs, desc=desc)


class Idx(object):
    '''
    Calculate the indices of values by interpolation in a LUT axis
    The index method is typically called when indexing a dimension of a LUT
    object by a Idx object.
    The round attribute (boolean) indicates whether the resulting index should
    be rounded to the closest integer.

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
    '''
    def __init__(self, value, name=None, round=False):
        if value is not None:
            self.value = value
            self.name = name
            self.round = round

    def index(self, axis):
        '''
        Return the floating point index of the values in the axis
        '''
        # axis is scalar or ndarray: interpolate
        res = interp1d(axis, np.arange(len(axis)))(self.value)

        if self.round:
            if isinstance(res, np.ndarray):
                res = res.round().astype(int)
            else:
                res = round(res)

        return res


def merge(luts, axes):
    '''
    Merge several LUTs or MLUTs, return the merged LUT or MLUT

    Arguments:
        - luts is a list of LUT or MLUT objects
        - axes is a list of axes names to merge
          these names should be present in each LUT attribute

    Example

    >>> np.random.seed(0)

    create four 2D look-up tables with identical axes and 2 attributes
    >>> ax1 = np.arange(4)     # 0..3
    >>> ax2 = np.arange(5)+10  # 10..14
    >>> L1 = LUT(np.random.rand(4, 5), axes=[ax1, ax2], names=['C','D'], attrs={'A':11, 'B': 1})
    >>> L2 = LUT(np.random.rand(4, 5), axes=[ax1, ax2], names=['C','D'], attrs={'A':11, 'B': 2})
    >>> Lmerged = merge([L1, L2], ['B'])  # merge 2 luts (1 new dimension 'B')
    >>> Lmerged.shape, Lmerged.attrs    # should contain 'A', which has not been merged
    ((2, 4, 5), {'A': 11})

    create 2 more LUTs and merge 4 luts over 2 dimensions
    merge them and create two new axes 'A' and 'B', using the attributes
    provided in each LUT attributes
    >>> L3 = LUT(np.random.rand(4, 5), axes=[ax1, ax2], names=['C','D'], attrs={'A':10, 'B': 1})
    >>> L4 = LUT(np.random.rand(4, 5), axes=[ax1, ax2], names=['C','D'], attrs={'A':10, 'B': 2})
    >>> Lmerged = merge([L1, L2, L3, L4], ['A', 'B'])
    >>> Lmerged.shape
    (2, 2, 4, 5)
    '''
    # merge MLUT
    if isinstance(luts[0], MLUT):
        M = []
        for p in luts[0].params:
            merged = merge(map(lambda m: m[p], luts), axes)
            M.append(merged)
        return MLUT(M)

    # verify lut axis and desc compatibility
    for lut in luts:
        for i in xrange(lut.ndim):
            a1 = lut.axes[i]
            a2 = luts[0].axes[i]
            if isinstance(a1, np.ndarray):
                assert (a1 == a2).all()
            else:
                assert a1 == a2

    # determine the new axes from the attributes of all luts
    N = len(axes)
    newaxes = []
    for _ in xrange(N):
        newaxes.append([])
    for lut in luts:
        for i in xrange(N):
            a = axes[i]
            value = lut.attrs[a]
            if not value in newaxes[i]:
                newaxes[i].append(value)

    new_shape = tuple(map(len, newaxes))
    new_shape += lut.shape

    # build new data
    newdata = np.zeros(new_shape)+np.NaN
    for lut in luts:
        # find the index of the attributes in the new LUT
        index = ()
        for i in xrange(N):
            a = axes[i]
            index += (newaxes[i].index(lut.attrs[a]),)
        index += (None,)

        newdata[index] = lut.data[:]

    # determine new names and attributes
    newnames = axes + luts[0].names
    newattrs = {}
    for a, v in luts[0].attrs.items():
        if not a in axes:
            # any attribute in the new LUT should be identical in all the
            # merged luts
            for lut in luts:
                if lut.attrs[a] != v:   # this attribute cannot be merged
                    continue
            newattrs[a] = v

    # convert list dimensions to ndarray if all elements are numeric
    for i in xrange(N):
        assert np.all(map(lambda x: isinstance(x, (int, float)), newaxes[i]))
        newaxes[i] = np.array(newaxes[i])

    # new axes: append old axes
    newaxes += luts[0].axes

    # include desc only if it is the same for all merged LUTS
    desc = luts[0].desc
    for lut in luts:
        if lut.desc != desc:
            desc = None
            break

    return LUT(newdata, axes=newaxes, names=newnames, attrs=newattrs, desc=desc)


class MLUT(object):
    '''
    A class to manage multiple Look-up tables (a list of LUTs)
    '''
    def __init__(self, luts):
        self.params = []
        self.luts = []
        for lut in luts:
            assert lut.desc is not None, 'please provide a desc'
            assert lut.desc not in self.params, 'lut.desc is not unique'
            self.params.append(lut.desc)
            self.luts.append(lut)

    def save(self, filename, overwrite=False):
        '''
        save a MLUT in a hdf file
        '''
        if exists(filename):
            if overwrite:
                remove(filename)
            else:
                ex = Exception('File {} exists'.format(filename))
                setattr(ex, 'filename', filename)
                raise ex

        for p in self.params:
            self.__getitem__(p).save(filename)

    def check_compatible(self, other):
        '''
        check that self is compatible with other for binary operation
        '''
        if isinstance(other, MLUT):
            assert (self.params == other.params)
            return other.luts
        else:
            return [other]*len(self.params)

    def __add__(self, other):
        otherlut = self.check_compatible(other)
        return MLUT(map(lambda x,y: x+y, self.luts, otherlut))

    def __radd__(self, other):
        return self+other

    def __sub__(self, other):
        otherlut = self.check_compatible(other)
        return MLUT(map(lambda x,y: x-y, self.luts, otherlut))

    def __rsub__(self, other):
        otherlut = self.check_compatible(other)
        return MLUT(map(lambda x,y: x-y, otherlut, self.luts))

    def __mul__(self, other):
        otherlut = self.check_compatible(other)
        return MLUT(map(lambda x,y: x*y, self.luts, otherlut))

    def __rmul__(self, other):
        return self*other

    def __div__(self, other):
        otherlut = self.check_compatible(other)
        return MLUT(map(lambda x,y: x/y, self.luts, otherlut))

    def __rdiv__(self, other):
        otherlut = self.check_compatible(other)
        return MLUT(map(lambda x,y: x/y, otherlut, self.luts))

    def print_info(self):
        print 'MLUT has {} parameters:'.format(len(self.params))
        for n in self.params:
            i = self.params.index(n)
            self.luts[i].print_info('  ')

    def __getitem__(self, key):
        if not key in self.params:
            raise Exception('{} is not present in MLUT'.format(key))
        i = self.params.index(key)
        return self.luts[i]


def read_mlut_hdf(filename, datasets=None, axnames=None):
    '''
    read datasets in filename, and return them as a LUTS
    datasets: list of datasets to read (default None, read all datasets having
    an attribute 'dimensions' or being compatible with axnames)
    axnames: override the attribute dimensions
    '''
    hdf = SD(filename)

    shape = []
    if axnames is not None:
        for d in axnames:
            (sdsname, rank, shp, dtype, nattr) = hdf.select(d).info()
            assert rank == 1
            shape.append(shp)

    if datasets is None:
        datasets = []
        for d in xrange(len(hdf.datasets())):
            sds = hdf.select(d)
            (sdsname, rank, shp, dtype, nattr) = sds.info()

            if ((axnames is not None) and (shape == shp)) or ('dimensions' in sds.attributes()):
                datasets.append(sdsname)

    luts = []
    for d in datasets:
        lut = read_lut_hdf(filename, d, axnames)
        luts.append(lut)

    return MLUT(luts)


def read_lut_hdf(filename, dataset, axnames=None):
    '''
    read a hdf file as a LUT, using axis list axnames
    if axnames is None, read the axes names in the attribute 'dimensions' of dataset
    '''
    axes = []
    names = []
    data = None
    dimensions = None

    hdf = SD(filename)

    # load dataset
    if dataset in hdf.datasets():
        sds = hdf.select(dataset)
        data = sds.get()
        if 'dimensions' in sds.attributes():
            dimensions = sds.attributes()['dimensions'].split(',')
            dimensions = map(lambda x: x.strip(), dimensions)
    else:
        print 'dataset "{}" not available.'.format(dataset)
        print '{} contains the following datasets:'.format(filename)
        for d in hdf.datasets():
            print '  *', d
        raise Exception('Missing dataset')

    if axnames == None:
        assert dimensions is not None, 'Error, dimension names have not been provided'
        axnames = dimensions
    else:
        if dimensions is not None:
            assert axnames == dimensions, 'Error in dimensions, expected {}, found {}'.format(axnames, dimensions)

    # load axes
    for d in axnames:
        sds = hdf.select(d)
        (sdsname, rank, shp, dtype, nattr) = sds.info()

        assert rank == 1

        axis = sds.get()
        axes.append(axis)
        names.append(sdsname)

    assert data.ndim == len(axes)
    for i in xrange(data.ndim):
        assert len(axes[i]) == data.shape[i]

    # read global attributes
    attrs = {}
    for a in hdf.attributes():
        attrs.update({a: hdf.attributes()[a]})

    return LUT(data, axes=axes, names=names, desc=dataset, attrs=attrs)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
