#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function, division, absolute_import

import numpy as np
from os import remove
from os.path import exists, join, basename
from warnings import warn

default_datasets = [
            'latitude', 'longitude',
            'Rw', 'Rnir', 'bitmask',
            'logchl', 'bbs', 'Rgli']
analysis_datasets = ['Rtoa', 'Rprime', 'Ratm', 'vza', 'sza', '_raa', 'niter', 'Rwmod', 'Tmol']
ancillary_datasets = ['ozone', 'surf_press', 'wind_speed', 'altitude']


class OutputExists(Exception):
    def __init__(self, filename):
        self.filename = filename
    def __str__(self):
        return 'File "{}" exists'.format(self.filename)


class Level2(object):
    '''
    Level2 initializer
    This class is responsible for creating a new level2 class in a context
    manager

    Arguments:
        fmt: format of level2. Can be:
                - hdf4 (default)
                - netcdf4
                - memory (returns a level2 object stored in memory)
        other kwargs are passed to the level2 object constructor
    '''
    def __init__(self, fmt='hdf4', **kwargs):
        if not 'ext' in kwargs:
            if fmt == 'hdf4':
                kwargs['ext'] = '.polymer.hdf'
            else:
                kwargs['ext'] = '.polymer.nc'

        self.kwargs = kwargs
        self.l2 = None

        if fmt == 'hdf4':
            from polymer.level2_hdf import Level2_HDF
            self.Level2 = Level2_HDF
        elif fmt == 'netcdf4':
            from polymer.level2_nc import Level2_NETCDF
            self.Level2 = Level2_NETCDF
        elif fmt == 'memory':
            from polymer.level2 import Level2_base
            self.Level2 = Level2_base
        else:
            raise Exception('Invalid format "{}"'.format(fmt))

    def __enter__(self):
        assert self.l2 is None

        # instantiate a new level2 object
        self.l2 = self.Level2(**self.kwargs)
        return self.l2

    def __exit__(self, type, value, traceback):
        self.l2.cleanup()
        self.l2 = None


class Level2_base(object):
    '''
    Base level 2 class (just store the product in memory)
    '''
    def __init__(self, datasets=None, **kwargs):
        self.datasets = datasets

    def init(self, level1):
        self.shape = level1.shape

        if self.datasets is None:
            self.datasets = default_datasets

    def write_block(self, name, data, S, attrs):

        if data.ndim == 2:
            if name not in self.__dict__:
                self.__dict__[name] = np.zeros(self.shape, dtype=data.dtype)
            self.__dict__[name][S] = data[:,:]

        elif data.ndim == 3:
            if name not in self.__dict__:
                self.__dict__[name] = np.zeros((self.shape+(len(self.bands),)), dtype=data.dtype)
            self.__dict__[name][S+(slice(None),)] = data[:,:,:]


    def write(self, block):
        assert self.shape is not None
        self.bands = block.bands

        (yoff, xoff) = block.offset
        (hei, wid) = block.size
        S = (slice(yoff,yoff+hei), slice(xoff,xoff+wid))

        to_write = list(self.datasets)

        for d in block.datasets():

            data = block[d]

            if not hasattr(data, 'ndim'):
                continue

            if (data.ndim == 2) and (d in self.datasets):
                to_write.remove(d)
                self.write_block(d, block[d], S,
                                 block.attributes.get(d, {}))

            if (data.ndim == 3):
                if d in self.datasets:
                    to_write.remove(d)
                    self.write_block(d, block[d], S,
                                     block.attributes.get(d, {}))
                else:
                    for iband, b in enumerate(self.bands):
                        dd = d+str(b) 
                        if dd in self.datasets:
                            to_write.remove(dd)
                            self.write_block(dd, block[d][:,:,iband], S,
                                             block.attributes.get(d, {}))

        if len(to_write) != 0:
            raise Exception('Error, could not find requested datasets: {}'.format(', '.join(to_write)))

    def attributes(self):
        return {}

    def finish(self, params):
        self.attrs = params.items()

    def cleanup(self):
        pass

class Level2_file(Level2_base):
    '''
    Base class for level 2 with file output
    '''
    def init(self, level1):

        self.shape = level1.shape

        assert level1.filename

        if self.filename is None:
            self.filename = level1.filename + self.ext
            if self.outdir is not None:
                self.filename = join(self.outdir, basename(self.filename))

        if exists(self.filename):
            if self.overwrite:
                print('Removing file', self.filename)
                remove(self.filename)
            else:
                raise OutputExists(self.filename)

        if self.datasets is None:
            self.datasets = default_datasets

        print('Initializing output file "{}"'.format(self.filename))





