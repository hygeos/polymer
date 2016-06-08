#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from pylab import imshow, colorbar, figure
from pyhdf.SD import SD, SDC
from luts import Idx
from os import remove
from os.path import exists
import warnings


class Level2(object):
    '''
    Base class level 2
    '''
    default_datasets = [
                'latitude', 'longitude',
                'Rtoa', 'vza', 'sza', 'raa',
                'Rprime',
                'Rw', 'Rnir', 'bitmask',
                'logchl', 'niter', 'Rgli']
    def init(self, level1):
        self.shape = level1.shape

class Level2_file(Level2):
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
                print 'Removing file', self.filename
                remove(self.filename)
            else:
                raise IOError('File "{}" exists'.format(self.filename))

        if self.datasets is None:
            self.datasets = self.default_datasets


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
    def __init__(self, filename, overwrite=False, datasets=None):

        self.filename = filename
        self.overwrite = overwrite
        self.datasets = datasets

        self.sdslist = {}
        self.typeconv = {
                    np.dtype('float32'): SDC.FLOAT32,
                    np.dtype('float64'): SDC.FLOAT64,
                    np.dtype('uint16'): SDC.UINT16,
                    np.dtype('uint32'): SDC.UINT32,
                    }

    def init(self, level1):
        super(self.__class__, self).init(level1)

        self.hdf = SD(self.filename, SDC.WRITE | SDC.CREATE)

    def write_block(self, name, data, S):
        '''
        write data into sds name with slice S
        '''

        # create dataset
        if name not in self.sdslist:
            dtype = self.typeconv[data.dtype]
            self.sdslist[name] = self.hdf.create(name, dtype, self.shape)

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
        for name, sds in self.sdslist.items():
            sds.endaccess()

        # write parameters
        warnings.warn('TODO')

        self.hdf.end()


class Level2_NETCDF(Level2):
    def __init__(self, filename, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        pass

    def write(self, block):
        pass

class Level2_Memory(Level2):
    '''
    Just store the product in memory
    '''
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



def contrast(x, max=1.):
    ''' stretch the contrast using a custom function '''
    R = np.sin(x/max*np.pi/2)**0.5
    R[R>max]=np.NaN
    return R

def RGB(data):
    figure()
    shp = (data.shape[1], data.shape[2], 3)
    RGB = np.zeros(shp)
    R = data[Idx(680, round=True), :, :]
    G = data[Idx(550, round=True), :, :]
    B = data[Idx(460, round=True), :, :]
    RGB[:,:,0] = contrast(R/np.amax(R[~np.isnan(R)]))
    RGB[:,:,1] = contrast(G/np.amax(G[~np.isnan(G)]))
    RGB[:,:,2] = contrast(B/np.amax(B[~np.isnan(B)]))
    imshow(RGB, interpolation='nearest')
    colorbar()

