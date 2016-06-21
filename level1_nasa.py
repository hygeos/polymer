#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from netCDF4 import Dataset
import numpy as np
from itertools import product
from block import Block


class Level1_NASA(object):
    '''
    Interface to NASA level-1C files
    Applies to sensors:
        - SeaWiFS
        - VIIRS
        - MODIS
    '''
    def __init__(self, filename, sensor=None, blocksize=(500, 400),
                 sline=0, eline=-1, srow=0, erow=-1):
        self.sensor = sensor
        self.filename = filename
        self.root = Dataset(filename)
        lat = self.root.groups['navigation_data'].variables['latitude']
        self.totalheight, self.totalwidth = lat.shape
        self.shape = self.totalheight, self.totalwidth
        self.sline = sline
        self.srow = srow
        self.blocksize = blocksize

        if eline < 0:
            self.height = self.totalheight
            self.height -= sline
            self.height += eline + 1
        else:
            self.height = eline-sline

        if erow < 0:
            self.width = self.totalwidth
            self.width -= srow
            self.width += erow + 1
        else:
            self.width = erow - srow


    def read_block(self, size, offset, bands):

        nbands = len(bands)
        size3 = size + (nbands,)

        SY = slice(offset[0]+self.sline, offset[0]+self.sline+size[0])
        SX = slice(offset[1]+self.srow , offset[1]+self.srow+size[1])

        # initialize block
        block = Block(offset=offset, size=size, bands=bands)

        # read lat/lon
        block.latitude = self.root.groups['navigation_data'].variables[
                'latitude'][SY, SX]
        block.longitude = self.root.groups['navigation_data'].variables[
                'longitude'][SY, SX]

        # read geometry
        block.sza = self.root.groups['geophysical_data'].variables['solz'][SY, SX]
        block.vza = self.root.groups['geophysical_data'].variables['senz'][SY, SX]
        block.saa = self.root.groups['geophysical_data'].variables['sola'][SY, SX]
        block.vaa = self.root.groups['geophysical_data'].variables['sena'][SY, SX]

        block.Rtoa = np.zeros(size3) + np.NaN
        for iband, band in enumerate(bands):
            Rtoa = self.root.groups['geophysical_data'].variables[
                    'rhot_{}'.format(band)][SY, SX]

            polcor = self.root.groups['geophysical_data'].variables[
                    'polcor_{}'.format(band)][SY, SX]

            block.Rtoa[:,:,iband] = Rtoa/polcor

        block.bitmask = np.zeros(size, dtype='uint16')

        block.ozone = np.zeros(size, dtype='float32') + 300.  # FIXME
        block.wind_speed = np.zeros(size, dtype='float32') + 5.  # FIXME
        block.surf_press = np.zeros(size, dtype='float32') + 1013.   # FIXME

        block.jday = 120  # FIXME

        block.wavelen = np.zeros(size3, dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = float(band)

        print('Read', block)

        return block


    def blocks(self, bands_read):

        nblocks_h = int(np.ceil(float(self.height)/self.blocksize[0]))
        nblocks_w = int(np.ceil(float(self.width)/self.blocksize[1]))

        for (iblock_h, iblock_w) in product(range(nblocks_h), range(nblocks_w)):

            # determine block size
            if iblock_h == nblocks_h-1:
                ysize = self.height-(nblocks_h-1)*self.blocksize[0]
            else:
                ysize = self.blocksize[0]
            if iblock_w == nblocks_w-1:
                xsize = self.width-(nblocks_w-1)*self.blocksize[1]
            else:
                xsize = self.blocksize[1]
            size = (ysize, xsize)

            # determine the block offset
            yoffset = iblock_h * self.blocksize[0]
            xoffset = iblock_w * self.blocksize[1]
            offset = (yoffset, xoffset)

            yield self.read_block(size, offset, bands_read)


class Level1_VIIRS(Level1_NASA):
    ''' Interface to VIIRS Level-1C '''
    def __init__(self, filename, blocksize=(500, 400)):
        super(self.__class__, self).__init__(
                filename, blocksize=blocksize, sensor='VIIRS')

class Level1_SeaWiFS(Level1_NASA):
    ''' Interface to SeaWiFS Level-1C '''
    def __init__(self, filename, blocksize=(500, 400)):
        super(self.__class__, self).__init__(
                filename, blocksize=blocksize, sensor='SeaWiFS')

class Level1_MODIS(Level1_NASA):
    ''' Interface to MODIS Level-1C '''
    def __init__(self, filename, blocksize=(500, 400)):
        super(self.__class__, self).__init__(
                filename, blocksize=blocksize, sensor='MODIS')

