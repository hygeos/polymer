#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
from netCDF4 import Dataset
import numpy as np
from itertools import product
from block import Block
from datetime import datetime
from ancillary import Provider
from common import L2FLAGS


class Level1_NASA(object):
    '''
    Interface to NASA level-1C files
    Applies to sensors:
        - SeaWiFS
        - VIIRS
        - MODIS
    '''
    def __init__(self, filename, sensor=None, blocksize=(500, 400),
                 sline=0, eline=-1, srow=0, erow=-1, provider=None):
        self.sensor = sensor
        self.filename = filename
        self.root = Dataset(filename)
        lat = self.root.groups['navigation_data'].variables['latitude']
        self.totalheight, self.totalwidth = lat.shape
        self.sline = sline
        self.srow = srow
        self.blocksize = blocksize
        self.ancillary_initialized = False
        if provider is None:
            self.provider = Provider()
        else:
            self.provider = provider


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

        self.shape = (self.height, self.width)


        # read flag meanings
        var = self.root.groups['geophysical_data'].variables['l2_flags']
        flags = list(var.getncattr('flag_masks'))
        meanings = str(var.getncattr('flag_meanings')).split()
        self.flag_meanings = dict(zip(meanings, flags))


    def init_ancillary(self):
        if not self.ancillary_initialized:
            self.ozone = self.provider.get('ozone', self.date())
            self.wind_speed = self.provider.get('wind_speed', self.date())
            self.surf_press = self.provider.get('surf_press', self.date())

            self.ancillary_initialized = True

    def read_block(self, size, offset, bands):
        self.init_ancillary()

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

        # bitmask
        block.bitmask = np.zeros(size, dtype='uint16')
        flags = self.root.groups['geophysical_data'].variables['l2_flags'][SY, SX]
        block.bitmask += L2FLAGS['LAND']*(flags & self.flag_meanings['LAND'] != 0).astype('uint16')

        block.ozone = self.ozone[block.latitude, block.longitude]
        block.wind_speed = self.wind_speed[block.latitude, block.longitude]
        block.surf_press = self.surf_press[block.latitude, block.longitude]

        block.jday = self.date().timetuple().tm_yday

        block.wavelen = np.zeros(size3, dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = float(band)

        print('Read', block)

        return block

    def date(self):
        try:
            return self.__date
        except:
            dstart = datetime.strptime(self.root.getncattr('time_coverage_start'),
                                      '%Y-%m-%dT%H:%M:%S.%fZ')
            dstop = datetime.strptime(self.root.getncattr('time_coverage_end'),
                                      '%Y-%m-%dT%H:%M:%S.%fZ')

            self.__date = dstart + (dstop - dstart)//2

            return self.__date


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
    def __init__(self, filename, **kwargs):
        super(self.__class__, self).__init__(
                filename, sensor='VIIRS', **kwargs)

class Level1_SeaWiFS(Level1_NASA):
    ''' Interface to SeaWiFS Level-1C '''
    def __init__(self, filename, **kwargs):
        super(self.__class__, self).__init__(
                filename, sensor='SeaWiFS', **kwargs)

class Level1_MODIS(Level1_NASA):
    ''' Interface to MODIS Level-1C '''
    def __init__(self, filename, **kwargs):
        super(self.__class__, self).__init__(
                filename, sensor='MODIS', **kwargs)

