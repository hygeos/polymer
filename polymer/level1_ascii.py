#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import numpy as np
import pandas as pd
from polymer.block import Block
from datetime import datetime
from os.path import join, dirname
from polymer.level1_meris import BANDS_MERIS
from polymer.common import L2FLAGS
from polymer.utils import raiseflag
from polymer.level1_meris import central_wavelength_meris
from polymer.level1_olci import central_wavelength_olci
import warnings

# bands stored in the ASCII extractions
BANDS_MODIS = [412,443,469,488,531,547,555,645,667,678,748,858,869,1240]
BANDS_SEAWIFS = [412,443,490,510,555,670,765,865]
BANDS_VIIRS = [410,443,486,551,671,745,862,1238,1601,2257]
BANDS_OLCI = [400 , 412, 443 , 490, 510 , 560, 620 , 665,
              674 , 681, 709 , 754, 760 , 764, 767 , 779,
              865 , 885, 900 , 940, 1020]

headers_default = {
                   'TOA': 'TOAR_{:02d}',
                   'F0': 'F0_{:02d}',
                   'LAMBDA0': 'LAMBDA0_{:02d}',
                   'LAT': 'LAT',
                   'LON': 'LON',
                   'DATETIME': 'TIME',
                   'DETECTOR_INDEX': 'DETECTOR',
                   'OZONE': 'OZONE_ECMWF',
                   'WIND': 'WINDM',
                   'SURFACE_PRESSURE': 'PRESS_ECMWF',
                   'ALTITUDE':'ALTITUDE',
                   'SZA': 'SUN_ZENITH',
                   'VZA': 'VIEW_ZENITH',
                   'RAA': 'DELTA_AZIMUTH',
                   }


class Level1_ASCII(object):
    '''
    Interface to ASCII data

    ascii file contains extractions of square x square pixels
    data are processed by blocks of blocksize

    arguments:
        * additional_headers (list of strings): additional datasets to read in
          the ASCII file and store in self.csv
        * TOAR: 'radiance' or 'reflectance'
        * relative_azimuth: boolean
        * wind_module: if True, read the wind module
                       else read the zonal and meridinal wind speeds
                       or float to use a constant value
        * headers: dictionary
    '''
    def __init__(self, filename, square=1, blocksize=100,
                 additional_headers=[], dir_smile=None,
                 sensor=None, BANDS=None, TOAR='radiance',
                 headers=headers_default,
                 relative_azimuth=True,
                 wind_module=True,
                 datetime_fmt='%Y%m%dT%H%M%SZ', verbose=True,
                 sep=';', skiprows=0):

        self.sensor = sensor
        self.filename = filename
        self.TOAR = TOAR
        self.headers = headers
        self.relative_azimuth = relative_azimuth
        self.wind_module = wind_module
        self.verbose = verbose

        if BANDS is None:
            BANDS = {
                    'MERIS': BANDS_MERIS,
                    'MERIS_RR': BANDS_MERIS,
                    'MERIS_FR': BANDS_MERIS,
                    'SeaWiFS': BANDS_SEAWIFS,
                    'MODIS': BANDS_MODIS,
                    'VIIRS': BANDS_VIIRS,
                    'OLCI': BANDS_OLCI
                    }[sensor]

        self.band_names = dict(map(lambda b: (b[1], self.headers['TOA'].format(b[0]+1)),
                                   enumerate(BANDS)))

        if sensor in ['MERIS', 'MERIS_RR', 'MERIS_FR']:
            if dir_smile is None:
                dir_smile = join(dirname(dirname(__file__)), 'auxdata/meris/smile/v2/')

            if sensor == 'MERIS_FR':
                self.F0 = np.genfromtxt(join(dir_smile, 'sun_spectral_flux_fr.txt'), names=True)
                self.detector_wavelength = np.genfromtxt(join(dir_smile, 'central_wavelen_fr.txt'), names=True)
            else:  # MERIS_RR or MERIS
                self.F0 = np.genfromtxt(join(dir_smile, 'sun_spectral_flux_rr.txt'), names=True)
                self.detector_wavelength = np.genfromtxt(join(dir_smile, 'central_wavelen_rr.txt'), names=True)

            self.F0_band_names = dict(map(lambda b: (b[1], 'E0_band{:d}'.format(b[0])),
                                          enumerate(BANDS)))
            self.wav_band_names = dict(map(lambda b: (b[1], 'lam_band{:d}'.format(b[0])),
                                           enumerate(BANDS)))
        
        #
        # read the csv file (only the required columns)
        #
        columns = []
        for c in ['LAT', 'LON', 'DATETIME',
                  'OZONE', 'SURFACE_PRESSURE','ALTITUDE',
                  'SZA', 'VZA']:
            columns.append(self.headers[c])
        if self.relative_azimuth:
            columns.append(self.headers['RAA'])
        else:
            columns.append(self.headers['SAA'])
            columns.append(self.headers['VAA'])

        if isinstance(self.wind_module, float):
            pass
        elif self.wind_module:
            columns.append(self.headers['WIND'])
        else:
            columns.append(self.headers['ZONAL_WIND'])
            columns.append(self.headers['MERID_WIND'])

        if sensor in ['MERIS', 'MERIS_RR', 'MERIS_FR']:
            columns.append(self.headers['DETECTOR_INDEX'])
        if 'F0' in self.headers:
            columns += map(lambda b: self.headers['F0'].format(b[0]+1), enumerate(BANDS))
        if 'LAMBDA0' in self.headers:
            columns += map(lambda b: self.headers['LAMBDA0'].format(b[0]+1), enumerate(BANDS))

        columns += additional_headers
        columns += self.band_names.values()
        if self.verbose:
            print('Reading from CSV file "{}"...'.format(filename))
            print('{} columns: {}'.format(len(columns), str(columns)))
        self.csv = pd.read_csv(filename,
                sep=sep,
                usecols = columns,
                skiprows=skiprows,
                )
        nrows = self.csv.shape[0]
        if self.verbose:
            print('Done (file has {} lines)'.format(nrows))
        assert nrows % square == 0
        self.height = nrows//square
        self.width = square
        self.shape = (self.height, self.width)
        self.blocksize = blocksize
        if self.verbose:
            print('Shape is', self.shape)

        self.dates = [datetime.strptime(x, datetime_fmt) for x in self.csv[self.headers['DATETIME']]]

    def get_field(self, fname, sl, size):
        cname = self.headers[fname]
        return self.csv[cname][sl].values.reshape(size).astype('float32')

    def read_block(self, size, offset, bands):

        (ysize, xsize) = size
        nbands = len(bands)

        # initialize block
        block = Block(offset=offset, size=size, bands=bands)
        sl = slice(offset[0]*xsize, (offset[0]+ysize)*xsize)
        block.wavelen = np.zeros((ysize,xsize,nbands), dtype='float32') + np.NaN
        block.cwavelen = np.zeros(nbands, dtype='float32') + np.NaN

        # coordinates
        block.latitude = self.get_field('LAT', sl, size)
        block.longitude = self.get_field('LON', sl, size)

        # read geometry
        block.sza = self.get_field('SZA', sl, size)
        block.vza = self.get_field('VZA', sl, size)
        if self.relative_azimuth:
            block._raa = self.get_field('RAA', sl, size)
        else:
            block.saa = self.get_field('SAA', sl, size)
            block.vaa = self.get_field('VAA', sl, size)

        # read TOA
        TOA = np.zeros((ysize,xsize,nbands)) + np.NaN
        for iband, band in enumerate(bands):
            name = self.band_names[band]
            TOA[:,:,iband] = self.csv[name][sl].values.reshape(size)

        if self.TOAR == 'reflectance':
            block.Rtoa = TOA

        elif self.TOAR == 'radiance':
            block.Ltoa = TOA

        else:
            raise Exception('Invalid TOAR type "{}"'.format(self.TOAR))

        # detector index
        if self.sensor in ['MERIS', 'MERIS_FR', 'MERIS_RR']:
            di = self.csv[self.headers['DETECTOR_INDEX']][sl].values.reshape(size).astype('int')

            # F0
            block.F0 = np.zeros((ysize, xsize, nbands)) + np.NaN
            for iband, band in enumerate(bands):
                block.F0[:,:,iband] = self.F0[self.F0_band_names[band]][di]

            # detector wavelength
            for iband, band in enumerate(bands):
                block.wavelen[:,:,iband] = self.detector_wavelength[self.wav_band_names[band]][di]
                block.cwavelen[iband] = central_wavelength_meris[band]
        elif self.sensor in ['OLCI']:
            block.F0 = np.zeros((ysize, xsize, nbands)) + np.NaN
            for iband, band in enumerate(bands):
                # F0
                name = self.headers['F0'].format(BANDS_OLCI.index(band)+1)
                block.F0[:,:,iband] = self.csv[name][sl].values.reshape(size)
                # detector wavelength
                name = self.headers['LAMBDA0'].format(BANDS_OLCI.index(band)+1)
                block.wavelen[:,:,iband] = self.csv[name][sl].values.reshape(size)
                block.cwavelen[iband] = float(band)#central_wavelength_olci[band]"""
        else:
            if 'F0' in self.headers:
                block.F0 = np.zeros((ysize, xsize, nbands)) + np.NaN

            for iband, band in enumerate(bands):
                block.wavelen[:,:,iband] = float(band)
                block.cwavelen[iband] = float(band)
                warnings.warn('Level1_ASCII does not properly take into account spectral information for this sensor.')
                # TODO
                # 1) check that F0 is seasonally corrected here
                # 2) properly read wavelen and cwavelen

                if 'F0' in self.headers:
                    name = self.headers['F0'].format(iband+1)
                    block.F0[:,:,iband] = self.csv[name][sl].values.reshape(size)

        block.jday = np.array([x.timetuple().tm_yday for x in self.dates[sl]]).reshape(size)

        block.month = np.array([x.timetuple().tm_mon for x in self.dates[sl]]).reshape(size)

        block.bitmask = np.zeros(size, dtype='uint16')
        invalid = np.isnan(block.raa)
        invalid |= TOA[:,:,0] < 0
        raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], invalid)

        # ozone
        block.ozone = self.csv[self.headers['OZONE']][sl].values.reshape(size)
        if self.sensor == 'OLCI':
            block.ozone /= 2.1415e-5  # convert kg/m2 to DU

        # wind speed
        if isinstance(self.wind_module, float):
            block.wind_speed = np.zeros(size)
            block.wind_speed[:] = self.wind_module
        elif self.wind_module:
            block.wind_speed = self.csv[self.headers['WIND']][sl].values.reshape(size)
        else:
            zwind = self.csv[self.headers['ZONAL_WIND']][sl].values.reshape(size)
            mwind = self.csv[self.headers['MERID_WIND']][sl].values.reshape(size)
            block.wind_speed = np.sqrt(zwind**2 + mwind**2)

        # surface pressure
        block.surf_press = self.csv[self.headers['SURFACE_PRESSURE']][sl].values.reshape(size)

        # altitude
        block.altitude = self.csv[self.headers['ALTITUDE']][sl].values.reshape(size)

        return block

    def blocks(self, bands_read):

        nblocks = int(np.ceil(float(self.height)/self.blocksize))
        for iblock in range(nblocks):

            # determine block size
            xsize = self.width
            if iblock == nblocks-1:
                ysize = self.height-(nblocks-1)*self.blocksize
            else:
                ysize = self.blocksize
            size = (ysize, xsize)

            # determine the block offset
            xoffset = 0
            yoffset = iblock*self.blocksize
            offset = (yoffset, xoffset)

            yield self.read_block(size, offset, bands_read)


    def attributes(self, datefmt):
        attr = {}
        attr['l1_filename'] = self.filename
        return attr

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

