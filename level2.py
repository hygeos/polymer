#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from pylab import imshow, show, colorbar, figure
from pyhdf.SD import SD, SDC
from luts import Idx

class Level2_HDF(object):
    def __init__(self, filename):
        self.filename = filename
        self.hdf = SD(filename, SDC.WRITE | SDC.CREATE)

    def init(self, level1):
        self.shape = level1.shape
        self.sds = self.hdf.create('logchl', SDC.FLOAT32, self.shape)

    def write(self, block):
        print 'write', block
        (yoff, xoff) = block.offset
        (hei, wid) = block.size

        self.sds[yoff:yoff+hei,xoff:xoff+wid] = block.logchl[:,:]

    def finish(self):
        self.sds.endaccess()
        self.hdf.end()


class Level2_NETCDF(object):

    def __init__(self, filename):
        pass

    def write(self, block):
        pass

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

class Level2_Memory(object):
    '''
    Just store the product in memory
    '''

    def init(self, level1, list_datasets=[
            'Rtoa', 'Rprime', 'Rnir', 'bitmask', 'logchl', 'niter']):
        self.shape = level1.shape
        self.list_datasets = list_datasets

    def write(self, block):

        (yoff, xoff) = block.offset
        (hei, wid) = block.size

        for d in self.list_datasets:

            data = block.__dict__[d]

            if data.ndim == 2:
                if d not in self.__dict__:
                    self.__dict__[d] = np.zeros(self.shape, dtype=data.dtype)
                self.__dict__[d][yoff:yoff+hei,xoff:xoff+wid] = data[:,:]

            elif data.ndim == 3:
                if d not in self.__dict__:
                    self.__dict__[d] = np.zeros(((len(block.bands),)+self.shape), dtype=data.dtype)
                self.__dict__[d][:,yoff:yoff+hei,xoff:xoff+wid] = data[:,:,:]

            else:
                raise Exception('Error')

    def finish(self):
        pass


