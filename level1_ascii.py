#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import pandas as pd
from block import Block
from datetime import datetime
from os.path import join
from os import getcwd


class Level1_ASCII(object):
    def __init__(self, filename, square=1, blocksize=100, additional_headers=[], dir_smile=None):
        '''
        Interface to ASCII data

        ascii file contains extractions of square x square pixels
        data are processed by blocks of blocksize
        additional_headers are also read in the ASCII file
        '''

        self.sensor = 'MERIS'

        if dir_smile is None:
            dir_smile = join(getcwd(), 'auxdata/meris/smile/v2/')

        self.band_names = {
                412: 'TOAR_01', 443: 'TOAR_02',
                490: 'TOAR_03', 510: 'TOAR_04',
                560: 'TOAR_05', 620: 'TOAR_06',
                665: 'TOAR_07', 681: 'TOAR_08',
                709: 'TOAR_09', 754: 'TOAR_10',
                760: 'TOAR_11', 779: 'TOAR_12',
                865: 'TOAR_13', 885: 'TOAR_14',
                900: 'TOAR_15',
            }

        # initialize solar irradiance
        self.F0 = np.genfromtxt(join(dir_smile, 'sun_spectral_flux_rr.txt'), names=True)
        self.F0_band_names = {
                    412: 'E0_band0', 443: 'E0_band1',
                    490: 'E0_band2', 510: 'E0_band3',
                    560: 'E0_band4', 620: 'E0_band5',
                    665: 'E0_band6', 681: 'E0_band7',
                    709: 'E0_band8', 754: 'E0_band9',
                    760: 'E0_band10', 779: 'E0_band11',
                    865: 'E0_band12', 885: 'E0_band13',
                    900: 'E0_band14',
                    }
        self.detector_wavelength = np.genfromtxt(join(dir_smile, 'central_wavelen_rr.txt'), names=True)
        self.wav_band_names = {
                    412: 'lam_band0', 443: 'lam_band1',
                    490: 'lam_band2', 510: 'lam_band3',
                    560: 'lam_band4', 620: 'lam_band5',
                    665: 'lam_band6', 681: 'lam_band7',
                    709: 'lam_band8', 754: 'lam_band9',
                    760: 'lam_band10', 779: 'lam_band11',
                    865: 'lam_band12', 885: 'lam_band13',
                    900: 'lam_band14',
                }

        #
        # read the csv file
        #
        columns = ['LAT', 'LON', 'TIME', 'DETECTOR']
        columns += ['OZONE_ECMWF', 'WINDM', 'PRESS_ECMWF']
        columns += ['SUN_ZENITH', 'VIEW_ZENITH', 'DELTA_AZIMUTH']
        columns += additional_headers
        columns += self.band_names.values()
        print('Reading CSV file "{}"...'.format(filename))
        self.csv = pd.read_csv(filename,
                sep=';',
                usecols = columns,
                )
        nrows = self.csv.shape[0]
        print('Done (file has {} lines)'.format(nrows))
        assert nrows % square == 0
        self.height = nrows//square
        self.width = square
        self.shape = (self.height, self.width)
        self.blocksize = blocksize
        print('Shape is', self.shape)

        self.dates = map(
                lambda x: datetime.strptime(x, '%Y%m%dT%H%M%SZ'),
                self.csv['TIME'])

    def get_field(self, fname, sl, size):
        return self.csv[fname][sl].reshape(size).astype('float32')

    def read_block(self, size, offset, bands):

        (ysize, xsize) = size
        nbands = len(bands)

        # initialize block
        block = Block(offset=offset, size=size, bands=bands)
        sl = slice(offset[0]*xsize, (offset[0]+ysize)*xsize)

        # coordinates
        block.latitude = self.get_field('LAT', sl, size)
        block.longitude = self.get_field('LON', sl, size)

        # read geometry
        block.sza = self.get_field('SUN_ZENITH', sl, size)
        block.vza = self.get_field('VIEW_ZENITH', sl, size)
        block._raa = self.get_field('DELTA_AZIMUTH', sl, size)

        # read TOA
        Ltoa = np.zeros((ysize,xsize,nbands)) + np.NaN
        for iband, band in enumerate(bands):
            name = self.band_names[band]
            Ltoa[:,:,iband] = self.csv[name][sl].reshape(size)
        block.Ltoa = Ltoa

        # detector index
        di = self.csv['DETECTOR'][sl].reshape(size)

        # F0
        block.F0 = np.zeros((ysize, xsize, nbands)) + np.NaN
        for iband, band in enumerate(bands):
            block.F0[:,:,iband] = self.F0[self.F0_band_names[band]][di]

        # detector wavelength
        block.wavelen = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = self.detector_wavelength[self.wav_band_names[band]][di]

        block.jday = np.array(map(lambda x: x.timetuple().tm_yday,
                                  self.dates[sl])).reshape(size)
        block.bitmask = np.zeros(size, dtype='uint16')

        # ozone
        block.ozone = self.csv['OZONE_ECMWF'][sl].reshape(size)

        # wind speed
        block.wind_speed = self.csv['WINDM'][sl].reshape(size)

        # surface pressure
        block.surf_press = self.csv['PRESS_ECMWF'][sl].reshape(size)

        return block

    def blocks(self, bands_read):

        nblocks = int(np.ceil(float(self.height)/self.blocksize))
        for iblock in xrange(nblocks):

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


    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

