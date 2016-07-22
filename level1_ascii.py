#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import pandas as pd
from block import Block
from datetime import datetime
from os.path import join
from os import getcwd
from level1_meris import BANDS_MERIS


class Level1_ASCII(object):
    '''
    Interface to ASCII data

    ascii file contains extractions of square x square pixels
    data are processed by blocks of blocksize
    additional_headers are also read in the ASCII file
    '''
    def __init__(self, filename, square=1, blocksize=100,
                 additional_headers=[], dir_smile=None,
                 sensor='MERIS'):

        self.sensor = sensor

        if dir_smile is None:
            dir_smile = join(getcwd(), 'auxdata/meris/smile/v2/')

        self.band_names = dict(map(lambda (i,b): (b, 'TOAR_{:02d}'.format(i+1)),
                                   enumerate(BANDS_MERIS)))

        # initialize solar irradiance
        self.F0 = np.genfromtxt(join(dir_smile, 'sun_spectral_flux_rr.txt'), names=True)
        self.F0_band_names = dict(map(lambda (i,b): (b, 'E0_band{:d}'.format(i)),
                                      enumerate(BANDS_MERIS)))
        self.detector_wavelength = np.genfromtxt(join(dir_smile, 'central_wavelen_rr.txt'), names=True)
        self.wav_band_names = dict(map(lambda (i,b): (b, 'lam_band{:d}'.format(i)),
                                       enumerate(BANDS_MERIS)))

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

