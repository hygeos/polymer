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
        488: 1.592E-01,
        531: 1.126E-01,
        547: 9.906E-02,
        667: 4.443E-02,
        678: 4.146E-02,
        748: 2.849E-02,
        869: 1.540E-02,
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

tau_r_seadas_viirs = {
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

def get_spectral_info(sensor):
    # read SRF to initialize effective wavelengths
    dir_auxdata = dirname(dirname(__file__))

    if sensor == 'MODIS':
        srf_file = join(dir_auxdata, 'auxdata/modisa/HMODISA_RSRs.txt')
        skiprows = 8
        bands = [412,443,469,488,531,547,555,645,667,678,748,858,869,1240,1640,2130]
        thres = 0.05
        tau_r_seadas = tau_r_seadas_modis
    elif sensor == 'SeaWiFS':
        srf_file = join(dir_auxdata, 'auxdata/seawifs/SeaWiFS_RSRs.txt')
        skiprows = 9
        bands = [412,443,490,510,555,670,765,865]
        thres = 0.2
        tau_r_seadas = tau_r_seadas_seawifs
    elif sensor == 'VIIRS':
        srf_file = join(dir_auxdata, 'auxdata/viirs/VIIRSN_IDPSv3_RSRs.txt')
        skiprows = 5
        bands = [410,443,486,551,671,745,862,1238,1601,2257]
        thres = 0.05
        tau_r_seadas = tau_r_seadas_viirs
    else:
        raise Exception('Invalid sensor "{}"'.format(sensor))

    srf = pd.read_csv(srf_file,
                      skiprows=skiprows, sep=None, engine='python',
                      skipinitialspace=True, header=None)

    central_wavelength = OrderedDict()
    for i, b in enumerate(bands):
        SRF = np.array(srf[i+1]).copy()
        SRF[SRF<thres] = 0.
        wav_eq = np.trapz(srf[0]*SRF)/np.trapz(SRF)
        central_wavelength[b] = wav_eq

    return central_wavelength, tau_r_seadas


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
                 altitude=0., use_srf=False):
        self.sensor = sensor
        self.filename = filename
        self.root = Dataset(filename)
        self.altitude = altitude
        self.use_srf = use_srf
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

        self.init_wavelengths()


    def init_ancillary(self):
        self.ozone = self.ancillary.get('ozone', self.date())
        self.wind_speed = self.ancillary.get('wind_speed', self.date())
        self.surf_press = self.ancillary.get('surf_press', self.date())

        self.ancillary_files = OrderedDict()
        self.ancillary_files.update(self.ozone.filename)
        self.ancillary_files.update(self.wind_speed.filename)
        self.ancillary_files.update(self.surf_press.filename)

    def init_wavelengths(self):
        # read SRF to initialize effective wavelengths

        self.central_wavelength, self.tau_r_seadas = get_spectral_info(self.sensor)


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
        block.saa = filled(saa[SY, SX])
        vaa = self.root.groups['geophysical_data'].variables['sena']
        vaa.set_auto_mask(False)
        block.vaa = filled(vaa[SY, SX])

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

        if not self.use_srf:
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

