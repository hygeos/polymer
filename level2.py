#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from pylab import imshow, show, colorbar, figure
from luts import Idx

class Level2_HDF(object):
    def __init__(self, filename):
        pass
    def write(self, data):
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

    def init(self, level1):
        self.shape = level1.shape
        self.Ltoa = None
        self.Rtoa = None
        self.Rprime = None
        self.sza = None
        self.bands = None

    def write(self, block):

        (xoff, yoff) = block.offset
        (wid, hei) = block.size

        if self.bands is None:
            self.bands = block.bands

        if self.sza is None:
            self.sza = np.zeros(self.shape) + np.NaN

        if self.Ltoa is None:
            self.Ltoa = np.zeros((len(block.bands),)+self.shape) + np.NaN

        if self.Rtoa is None:
            self.Rtoa = np.zeros((len(block.bands),)+self.shape) + np.NaN

        if self.Rprime is None:
            self.Rprime = np.zeros((len(block.bands),)+self.shape) + np.NaN

        self.Ltoa[:,yoff:yoff+hei,xoff:xoff+wid] = block.Ltoa[:,:,:]
        self.Rtoa[:,yoff:yoff+hei,xoff:xoff+wid] = block.Rtoa[:,:,:]
        self.Rprime[:,yoff:yoff+hei,xoff:xoff+wid] = block.Rprime[:,:,:]
        self.sza[yoff:yoff+hei,xoff:xoff+wid] = block.sza[:,:]

    def finish(self):
        pass


