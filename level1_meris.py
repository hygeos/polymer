#!/usr/bin/env python
# encoding: utf-8


import epr
from block import Block
import numpy as np

class Level1_MERIS(object):

    def __init__(self, filename):

        self.prod = epr.Product(filename)
        self.width = self.prod.get_scene_width()
        self.height = self.prod.get_scene_height()
        self.shape = (self.height, self.width)
        self.band_names = {
                412: 'Radiance_1', 443: 'Radiance_2',
                490: 'Radiance_3', 510: 'Radiance_4',
                560: 'Radiance_5', 620: 'Radiance_6',
                665: 'Radiance_7', 681: 'Radiance_8',
                709: 'Radiance_9', 754: 'Radiance_10',
                760: 'Radiance_11', 779: 'Radiance_12',
                865: 'Radiance_13', 885: 'Radiance_14',
                900: 'Radiance_15',
            }

        # initialize solar irradiance
        self.F0 = np.genfromtxt('/home/francois/MERIS/POLYMER/auxdata/meris/smile/v2/sun_spectral_flux_rr.txt', names=True)
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

        # initialize detector wavelength
        self.detector_wavelength = np.genfromtxt('/home/francois/MERIS/POLYMER/auxdata/meris/smile/v2/central_wavelen_rr.txt', names=True)

        print 'Opened "{}", ({}x{})'.format(filename, self.width, self.height)

    def read_band(self, band_name, size, offset):
        (xsize, ysize) = size
        (xoffset, yoffset) = offset
        return self.prod.get_band(band_name).read_as_array(
                    xoffset=xoffset, yoffset=yoffset,
                    width=xsize, height=ysize)

    def blocks(self, bands_read=None, blocksize=100):

        nbands = len(bands_read)
        nblocks = self.height/blocksize + 1
        for iblock in xrange(nblocks):

            # determine block size
            xsize = self.width
            if iblock == nblocks-1:
                ysize = self.height-(nblocks-1)*blocksize
            else:
                ysize = blocksize
            size = (xsize, ysize)

            # determine the block offset
            xoffset = 0
            yoffset = iblock*blocksize
            offset = (xoffset, yoffset)

            # initialize block
            block = Block(offset=offset, size=size, bands=bands_read)

            # read geometry
            block.sza = self.read_band('sun_zenith', size, offset)
            block.vza = self.read_band('view_zenith', size, offset)
            block.saa = self.read_band('sun_azimuth', size, offset)
            block.vaa = self.read_band('view_azimuth', size, offset)

            # read detector index
            di = self.read_band('detector_index', size, offset)

            # calculate F0 for each band
            block.F0 = np.zeros((nbands, ysize, xsize)) + np.NaN
            for iband, band in enumerate(bands_read):
                block.F0[iband,:,:] = self.F0[self.F0_band_names[band]][di]

            # calculate detector wavelength for each band
            block.wavelen = np.zeros((nbands, ysize, xsize), dtype='float32') + np.NaN
            for iband, band in enumerate(bands_read):
                block.wavelen[iband,:,:] = self.detector_wavelength[self.wav_band_names[band]][di]

            # read TOA
            Ltoa = np.zeros((nbands, ysize, xsize)) + np.NaN
            for iband, band in enumerate(bands_read):
                Ltoa_ = self.read_band(self.band_names[band], size, offset)
                Ltoa[iband,:,:] = Ltoa_[:,:]
            block.Ltoa = Ltoa

            # wind speed (zonal and merdional)
            zwind = self.read_band('zonal_wind', size, offset)
            mwind = self.read_band('merid_wind', size, offset)
            block.wind_speed = np.sqrt(zwind**2 + mwind**2)

            print 'Reading', block

            yield block

