#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from netCDF4 import Dataset
import numpy as np
from polymer.block import Block
from datetime import datetime
from polymer.ancillary import Ancillary_NASA
from polymer.common import L2FLAGS
from collections import OrderedDict
from polymer.utils import raiseflag
from polymer.level1 import Level1_base


class Level1_NASA(Level1_base):
    '''
    Interface to NASA level-1C files
    Applies to sensors:
        - SeaWiFS
        - VIIRS
        - MODIS
    '''
    def __init__(self, filename, sensor=None, blocksize=(500, 400),
                 sline=0, eline=-1, scol=0, ecol=-1, ancillary=None):
        self.sensor = sensor
        self.filename = filename
        self.root = Dataset(filename)
        lat = self.root.groups['navigation_data'].variables['latitude']
        totalheight, totalwidth = lat.shape
        self.blocksize = blocksize
        if ancillary is None:
            self.ancillary = Ancillary_NASA()
        else:
            self.ancillary = ancillary

        self.init_shape(
                totalheight=totalheight,
                totalwidth=totalwidth,
                sline=sline,
                eline=eline,
                scol=scol,
                ecol=ecol)

        # read flag meanings
        var = self.root.groups['geophysical_data'].variables['l2_flags']
        flags = list(var.getncattr('flag_masks'))
        meanings = str(var.getncattr('flag_meanings')).split()
        self.flag_meanings = dict(zip(meanings, flags))

        # init dates
        self.__read_date()

        # initialize ancillary data
        self.init_ancillary()


    def init_ancillary(self):
        self.ozone = self.ancillary.get('ozone', self.date())
        self.wind_speed = self.ancillary.get('wind_speed', self.date())
        self.surf_press = self.ancillary.get('surf_press', self.date())

        self.ancillary_files = OrderedDict()
        self.ancillary_files.update(self.ozone.filename)
        self.ancillary_files.update(self.wind_speed.filename)
        self.ancillary_files.update(self.surf_press.filename)


    def read_block(self, size, offset, bands):

        nbands = len(bands)
        size3 = size + (nbands,)

        SY = slice(offset[0]+self.sline, offset[0]+self.sline+size[0])
        SX = slice(offset[1]+self.scol , offset[1]+self.scol+size[1])

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
        raiseflag(block.bitmask, L2FLAGS['LAND'],
                  flags & self.flag_meanings['LAND'] != 0)

        ok = block.latitude > -90.
        raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], ~ok)

        block.ozone = np.zeros_like(ok, dtype='float32')
        block.ozone[ok] = self.ozone[block.latitude[ok], block.longitude[ok]]
        block.wind_speed = np.zeros_like(ok, dtype='float32')
        block.wind_speed[ok] = self.wind_speed[block.latitude[ok], block.longitude[ok]]
        block.surf_press = np.zeros_like(ok, dtype='float32')
        block.surf_press[ok] = self.surf_press[block.latitude[ok], block.longitude[ok]]

        block.jday = self.date().timetuple().tm_yday
        block.month = self.date().timetuple().tm_mon

        block.wavelen = np.zeros(size3, dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = float(band)

        return block

    def __read_date(self):
        try:
            dstart = datetime.strptime(self.root.getncattr('time_coverage_start'),
                                  '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError: # try again without decimal part
            dstart = datetime.strptime(self.root.getncattr('time_coverage_start'),
                                  '%Y-%m-%dT%H:%M:%S')
        try:
            dstop = datetime.strptime(self.root.getncattr('time_coverage_end'),
                                  '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError: # try again without decimal part
            dstop = datetime.strptime(self.root.getncattr('time_coverage_end'),
                                  '%Y-%m-%dT%H:%M:%S')

        self.dstart = dstart
        self.dstop = dstop

    def date(self):
        return self.dstart + (self.dstop - self.dstart)//2

    def attributes(self, datefmt):
        '''
        Returns level1 attributes

        dates are formatted to string using datefmt
        '''
        attr = OrderedDict()
        attr['l1_filename'] = self.filename
        attr['start_time'] = self.dstart.strftime(datefmt)
        attr['stop_time'] = self.dstop.strftime(datefmt)

        attr.update(self.ancillary_files)

        return attr

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


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

