#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from level2 import Level2_file
from pyhdf.SD import SD, SDC
import numpy as np
from os import remove
from os.path import exists


class Level2_HDF(Level2_file):
    '''
    Level 2 in HDF4 format

    filename: string or function
        function taking level1 filename as input, returning a string
    overwrite: boolean
        overwrite existing file
    datasets: list or None
        list of datasets to include in level 2
        if None (default), use Level2.default_datasets
    '''
    def __init__(self, filename, overwrite=False, datasets=None, compress=True):

        self.filename = filename
        self.overwrite = overwrite
        self.datasets = datasets
        self.compress = True

        self.sdslist = {}
        self.typeconv = {
                    np.dtype('float32'): SDC.FLOAT32,
                    np.dtype('float64'): SDC.FLOAT64,
                    np.dtype('uint16'): SDC.UINT16,
                    np.dtype('uint32'): SDC.UINT32,
                    }

    def init(self, level1):
        super(self.__class__, self).init(level1)

        if not self.compress:
            self.__hdf = SD(self.filename, SDC.WRITE | SDC.CREATE)
        else:
            # one hdf object per dataset
            # which are merged upon finish()
            self.__hdf = {}

        yield

        # TODO: cleanup here

    def hdf(self, name):
        '''
        returns a hdf4 object for a given dataset name
        '''
        if not self.compress:
            return self.__hdf
        else:
            if not name in self.__hdf:
                filename = '{}_{}.tmp'.format(self.filename, name)
                if exists(filename):
                    print('Removing file', filename)
                    remove(filename)
                self.__hdf[name] = SD(filename, SDC.WRITE | SDC.CREATE)

            return self.__hdf[name]


    def write_block(self, name, data, S):
        '''
        write data into sds name with slice S
        '''

        # create dataset
        if name not in self.sdslist:
            dtype = self.typeconv[data.dtype]
            self.sdslist[name] = self.hdf(name).create(name, dtype, self.shape)

        # write
        self.sdslist[name][S] = data[:,:]


    def write(self, block):

        (yoff, xoff) = block.offset
        (hei, wid) = block.size
        S = (slice(yoff,yoff+hei), slice(xoff,xoff+wid))

        for d in self.datasets:

            # don't write dataset if not in block
            if d not in block.datasets():
                continue

            if block[d].ndim == 2:
                self.write_block(d, block[d], S)

            elif block[d].ndim == 3:
                for i, b in enumerate(block.bands):
                    sdsname = '{}{}'.format(d, b)
                    self.write_block(sdsname, block[d][:,:,i], S)
            else:
                raise Exception('Error ndim')

    def finish(self, params):
        if self.compress:
            hdf = SD(self.filename, SDC.WRITE | SDC.CREATE)

            for name in sorted(self.sdslist):
                print('Write compressed dataset {}'.format(name))
                sds = self.sdslist[name]

                dtype  = sds.info()[3]
                sds2 = hdf.create(name, dtype, self.shape)
                sds2.setcompress(SDC.COMP_DEFLATE, 9)
                sds2[:] = sds[:]

                sds2.endaccess()

        else:
            hdf = self.__hdf

        for name, sds in self.sdslist.items():
            sds.endaccess()

        # write attributes
        setattr(hdf, 'Test attribute', 1)

        if self.compress:
            # cleanup
            for name in self.__hdf:
                filename = '{}_{}.tmp'.format(self.filename, name)
                self.__hdf[name].end()
                remove(filename)
        else:
            hdf.end()


