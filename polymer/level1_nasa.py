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
from os.path import dirname, join
import pandas as pd


# Rayleigh optical thicknesses as defined in SeaDAS
tau_r_seadas_modis = {
        412: 3.099E-01,
        443: 2.367E-01,
        469: 1.914E-01,
        488: 1.592E-01,
        531: 1.126E-01,
        547: 9.906E-02,
        555: 9.432E-02,
        645: 5.082E-02,
        667: 4.443E-02,
        678: 4.146E-02,
        748: 2.849E-02,
        859: 1.613E-02,
        869: 1.540E-02,
        1240: 3.617E-03,
        }

tau_r_seadas_seawifs = {
        412: 3.128E-01,
        443: 2.329E-01,
        490: 1.542E-01,
        510: 1.326E-01,
        555: 9.444E-02,
        670: 4.444E-02,
        765: 2.553E-02,
        865: 1.690E-02,
        }

tau_r_seadas_viirsn = {
        410 : 3.175E-01,
        443 : 2.328E-01,
        486 : 1.600E-01,
        551 : 9.738E-02,
        671 : 4.395E-02,
        745 : 2.865E-02,
        862 : 1.594E-02,
        1238: 3.650E-03,
        1601: 1.305E-03,
        2257: 3.294E-04,
        }

tau_r_seadas_viirsj1 = {
        411 : 3.210E-01,
        445 : 2.312E-01,
        489 : 1.573E-01,
        556 : 9.252E-02,
        667 : 4.420E-02,
        746 : 2.815E-02,
        868 : 1.533E-02,
        1238: 3.650E-03,
        1604: 1.296E-03,
        2258: 3.285E-04,
        }


def filled(A, ok=None, fill_value=0):
    """
    Returns a filled from a filled or masked array, use fill_value
    modifies ok (if provided) to take this mask into account
    """
    if hasattr(A, 'filled'):
        # masked array: returns filled array
        if ok is not None:
            ok &= ~A.mask
        return A.filled(fill_value=fill_value)
    else:
        # filled array: does nothing
        return A


class Level1_NASA(Level1_base):
    '''
    Interface to NASA level-1C files
    Applies to sensors:
        - SeaWiFS
        - VIIRS
        - MODIS
    '''
    def __init__(self, filename, sensor=None, blocksize=(500, 400),
                 sline=0, eline=-1, scol=0, ecol=-1, ancillary=None,
                 altitude=0.):
        self.sensor = sensor
        self.filename = filename
        self.root = Dataset(filename)
        self.altitude = altitude
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

        self.init_ancillary()

        self.init_spectral_info()


    def init_ancillary(self):
        self.ozone = self.ancillary.get('ozone', self.date())
        self.wind_speed = self.ancillary.get('wind_speed', self.date())
        self.surf_press = self.ancillary.get('surf_press', self.date())

        self.ancillary_files = OrderedDict()
        self.ancillary_files.update(self.ozone.filename)
        self.ancillary_files.update(self.wind_speed.filename)
        self.ancillary_files.update(self.surf_press.filename)

    def init_spectral_info(self):
        # NOTE: central wavelengths are from SeaDAS

        if self.sensor == 'MODIS':
            bands = [412,443,469,488,531,547,555,645,667,678,748,859,869,1240,1640,2130]
            self.tau_r_seadas = tau_r_seadas_modis
        elif self.sensor == 'SeaWiFS':
            self.tau_r_seadas = tau_r_seadas_seawifs
            bands = [412,443,490,510,555,670,765,865]
        elif self.sensor in ['VIIRS', 'VIIRSN']:
            self.tau_r_seadas = tau_r_seadas_viirsn
            bands = [410,443,486,551,671,745,862,1238,1601,2257]
        elif self.sensor == 'VIIRSJ1':
            self.tau_r_seadas = tau_r_seadas_viirsj1
            bands = [411,445,489,556,667,746,868,1238,1604,2258]
        else:
            raise Exception('Invalid sensor "{}"'.format(self.sensor))

        self.central_wavelength = dict([(b, float(b)) for b in bands])


    def read_block(self, size, offset, bands):

        nbands = len(bands)
        size3 = size + (nbands,)
        (ysize, xsize) = size

        SY = slice(offset[0]+self.sline, offset[0]+self.sline+size[0])
        SX = slice(offset[1]+self.scol , offset[1]+self.scol+size[1])

        # initialize block
        block = Block(offset=offset, size=size, bands=bands)

        # read lat/lon
        block.latitude = filled(self.root.groups['navigation_data'].variables[
                'latitude'][SY, SX], fill_value=-999.)
        block.longitude = filled(self.root.groups['navigation_data'].variables[
                'longitude'][SY, SX], fill_value=-999.)

        ok = block.latitude > -90.

        # read geometry
        # note: we disactivate automasking because of bad formatting of SeaWiFS L1C, for which azimuth angles >180 are masked
        block.sza = filled(self.root.groups['geophysical_data'].variables['solz'][SY, SX], ok=ok)
        block.vza = filled(self.root.groups['geophysical_data'].variables['senz'][SY, SX], ok=ok)
        saa = self.root.groups['geophysical_data'].variables['sola']
        saa.set_auto_mask(False)
        block.saa = filled(saa[SY, SX]) % 360
        vaa = self.root.groups['geophysical_data'].variables['sena']
        vaa.set_auto_mask(False)
        block.vaa = filled(vaa[SY, SX]) % 360

        block.Rtoa = np.zeros(size3) + np.NaN
        for iband, band in enumerate(bands):
            Rtoa = filled(self.root.groups['geophysical_data'].variables[
                    'rhot_{}'.format(band)][SY, SX], ok=ok)

            polcor = filled(self.root.groups['geophysical_data'].variables[
                    'polcor_{}'.format(band)][SY, SX], ok=ok)

            block.Rtoa[:,:,iband] = Rtoa/polcor

        # bitmask
        block.bitmask = np.zeros(size, dtype='uint16')
        flags = filled(self.root.groups['geophysical_data'].variables['l2_flags'][SY, SX])
        raiseflag(block.bitmask, L2FLAGS['LAND'],
                flags & self.flag_meanings['LAND'] != 0)

        ok &= block.Rtoa[:,:,0] >= 0
        raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], ~ok)

        block.ozone = np.zeros_like(ok, dtype='float32')
        block.ozone[ok] = self.ozone[block.latitude[ok], block.longitude[ok]]
        block.wind_speed = np.zeros_like(ok, dtype='float32')
        block.wind_speed[ok] = self.wind_speed[block.latitude[ok], block.longitude[ok]]
        P0 = np.zeros_like(ok, dtype='float32')
        P0[ok] = self.surf_press[block.latitude[ok], block.longitude[ok]]

        # read surface altitude
        try:
            block.altitude = self.altitude.get(lat=block.latitude,
                                               lon=block.longitude)
        except AttributeError:
            # altitude expected to be a float
            block.altitude = np.zeros((ysize, xsize), dtype='float32') + self.altitude

        # calculate surface altitude
        block.surf_press = P0 * np.exp(-block.altitude/8000.)

        block.jday = self.date().timetuple().tm_yday
        block.month = self.date().timetuple().tm_mon

        block.wavelen = np.zeros(size3, dtype='float32') + np.NaN
        block.cwavelen = np.zeros(nbands, dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = self.central_wavelength[band]
            block.cwavelen[iband] = self.central_wavelength[band]

        block.tau_ray = np.zeros(size3, dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.tau_ray[:,:,iband] = self.tau_r_seadas[band] * block.surf_press/1013.

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
        attr['central_wavelength'] = self.central_wavelength

        attr.update(self.ancillary_files)

        return attr

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Level1_VIIRS(Level1_NASA):
    ''' Interface to VIIRS Level-1C '''
    def __init__(self, filename, **kwargs):
        root = Dataset(filename)
        platform = root.getncattr('platform')
        sensor = {
                'Suomi-NPP':'VIIRSN',
                'JPSS-1': 'VIIRSJ1',
                }[platform]
        super(self.__class__, self).__init__(
                filename, sensor=sensor, **kwargs)

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

