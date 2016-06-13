#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function

import numpy as np
from os import remove
from os.path import exists


class Level2_base(object):
    '''
    Base level 2 class (just store the product in memory)
    '''
    default_datasets = [
                'latitude', 'longitude',
                'Rtoa', 'vza', 'sza', 'raa',
                'Rprime',
                'Rw', 'Rnir', 'bitmask',
                'logchl', 'niter', 'Rgli']
    def __init__(self, datasets=None):
        self.datasets = datasets

    def init(self, level1):
        self.shape = level1.shape

        if self.datasets is None:
            self.datasets = self.default_datasets

    def write(self, block):
        assert self.shape is not None

        (yoff, xoff) = block.offset
        (hei, wid) = block.size

        for d in self.datasets:
            if d not in block.datasets():
                continue

            data = block[d]

            if data.ndim == 2:
                if d not in self.__dict__:
                    self.__dict__[d] = np.zeros(self.shape, dtype=data.dtype)
                self.__dict__[d][yoff:yoff+hei,xoff:xoff+wid] = data[:,:]

            elif data.ndim == 3:
                if d not in self.__dict__:
                    self.__dict__[d] = np.zeros((self.shape+(len(block.bands),)), dtype=data.dtype)
                self.__dict__[d][yoff:yoff+hei,xoff:xoff+wid,:] = data[:,:,:]

            else:
                raise Exception('Error')

    def finish(self, params):
        pass

class Level2_file(Level2_base):
    '''
    Base class for level 2 with file output
    '''
    def init(self, level1):

        self.shape = level1.shape

        assert level1.filename

        if not isinstance(self.filename, str):
            self.filename = self.filename(level1.filename)

        if exists(self.filename):
            if self.overwrite:
                print('Removing file', self.filename)
                remove(self.filename)
            else:
                raise IOError('File "{}" exists'.format(self.filename))

        if self.datasets is None:
            self.datasets = self.default_datasets

        print('Initializing output file "{}"'.format(self.filename))





