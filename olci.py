#!/usr/bin/env python
# -*- coding: utf-8 -*-

from params import Params
import numpy as np
from block import Block
import snappy
import warnings
from common import L2FLAGS
from netCDF4 import Dataset
from scipy.ndimage import map_coordinates
from datetime import datetime
import os


class Params_OLCI(Params):

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()

        # FIXME
        self.bands_corr = [412,443,490,510,560,620,665,754,779,865]
        self.bands_oc   = [412,443,490,510,560,620,665,754,779,865]
        self.bands_rw   = [412,443,490,510,560,620,665,754,779,865]

        self.bands_lut = [400,412,443,490,510,560,620,665,674,681,
                          709,754,760,764,767,779,865,885,900,940,
                          1020,1375,1610,2250]

        self.lut_file = '/home/francois/MERIS/POLYMER/LUTS/OLCI/LUT.hdf'

        self.band_cloudmask = 865

        # central wavelength of the detector where the Rayleigh optical thickness is calculated
        # (detector 374 of camera 3)
        self.central_wavelength = {
                400 : 400.664  , 412 : 412.076 ,
                443 : 443.183  , 490 : 490.713 ,
                510 : 510.639  , 560 : 560.579 ,
                620 : 620.632  , 665 : 665.3719,
                674 : 674.105  , 681 : 681.66  ,
                709 : 709.1799 , 754 : 754.2236,
                760 : 761.8164 , 764 : 764.9075,
                767 : 767.9734 , 779 : 779.2685,
                865 : 865.4625 , 885 : 884.3256,
                900 : 899.3162 , 940 : 939.02  ,
                1020: 1015.9766, 1375: 1375.   ,
                1610: 1610.    , 2250: 2250.   ,
                }

        self.K_OZ = {  # FIXME: taken from MERIS
                    412: 0.000301800 , 443: 0.00327200 ,
                    490: 0.0211900   , 510: 0.0419600  ,
                    560: 0.104100    , 620: 0.109100   ,
                    665: 0.0511500   , 681: 0.0359600  ,
                    709: 0.0196800   , 754: 0.00955800 ,
                    760: 0.00730400  , 779: 0.00769300 ,
                    865: 0.00219300  , 885: 0.00121100 ,
                    900: 0.00151600  ,
                }

        self.K_NO2 = {  # FIXME: taken from MERIS
                412: 6.074E-19 , 443: 4.907E-19,
                490: 2.916E-19 , 510: 2.218E-19,
                560: 7.338E-20 , 620: 2.823E-20,
                665: 6.626E-21 , 681: 6.285E-21,
                709: 4.950E-21 , 754: 1.384E-21,
                760: 4.717E-22 , 779: 3.692E-22,
                865: 2.885E-23 , 885: 4.551E-23,
                900: 5.522E-23 ,
                }

        self.update(**kwargs)


class Level1_OLCI_snappy(object):
    '''
    snappy reader
    '''

    def __init__(self, filename, sline=0, eline=-1):

        self.filename = filename

        self.prod = snappy.ProductIO.readProduct(filename)

        lat = self.prod.getBand('latitude')
        self.totalwidth = int(lat.getRasterWidth())
        self.width = self.totalwidth
        self.totalheight = int(lat.getRasterHeight())
        self.blocksize = 400

        self.sline = sline
        self.eline = eline
        if eline < 0:
            self.height = self.totalheight
            self.height -= sline
            self.height += eline + 1
        else:
            self.height = eline-sline

        self.shape = (self.height, self.width)


        self.band_names = {
                400 : 'Oa01_radiance', 412 : 'Oa02_radiance',
                443 : 'Oa03_radiance', 490 : 'Oa04_radiance',
                510 : 'Oa05_radiance', 560 : 'Oa06_radiance',
                620 : 'Oa07_radiance', 665 : 'Oa08_radiance',
                674 : 'Oa09_radiance', 681 : 'Oa10_radiance',
                709 : 'Oa11_radiance', 754 : 'Oa12_radiance',
                760 : 'Oa13_radiance', 764 : 'Oa14_radiance',
                767 : 'Oa15_radiance', 779 : 'Oa16_radiance',
                865 : 'Oa17_radiance', 885 : 'Oa18_radiance',
                900 : 'Oa19_radiance', 940 : 'Oa20_radiance',
                1020: 'Oa21_radiance',
                }

        self.F0_band_names = {
                400 : 'solar_flux_band_1' , 412 : 'solar_flux_band_2' ,
                443 : 'solar_flux_band_3' , 490 : 'solar_flux_band_4' ,
                510 : 'solar_flux_band_5' , 560 : 'solar_flux_band_6' ,
                620 : 'solar_flux_band_7' , 665 : 'solar_flux_band_8' ,
                674 : 'solar_flux_band_9' , 681 : 'solar_flux_band_10',
                709 : 'solar_flux_band_11', 754 : 'solar_flux_band_12',
                760 : 'solar_flux_band_13', 764 : 'solar_flux_band_14',
                767 : 'solar_flux_band_15', 779 : 'solar_flux_band_16',
                865 : 'solar_flux_band_17', 885 : 'solar_flux_band_18',
                900 : 'solar_flux_band_19', 940 : 'solar_flux_band_20',
                1020: 'solar_flux_band_21',
                }

        self.lambda0_band_names = {
                400 : 'lambda0_band_1' , 412 : 'lambda0_band_2' ,
                443 : 'lambda0_band_3' , 490 : 'lambda0_band_4' ,
                510 : 'lambda0_band_5' , 560 : 'lambda0_band_6' ,
                620 : 'lambda0_band_7' , 665 : 'lambda0_band_8' ,
                674 : 'lambda0_band_9' , 681 : 'lambda0_band_10',
                709 : 'lambda0_band_11', 754 : 'lambda0_band_12',
                760 : 'lambda0_band_13', 764 : 'lambda0_band_14',
                767 : 'lambda0_band_15', 779 : 'lambda0_band_16',
                865 : 'lambda0_band_17', 885 : 'lambda0_band_18',
                900 : 'lambda0_band_19', 940 : 'lambda0_band_20',
                1020: 'lambda0_band_21',
                }


    def read_band(self, band_name, size, offset, tiepoint=False):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset

        if tiepoint:
            rad = self.prod.getTiePointGrid(band_name)
        else:
            rad = self.prod.getBand(band_name)

        Ltoa_data = np.zeros((ysize,xsize), dtype='float32') + np.NaN
        rad.readPixels(xoffset, yoffset+self.sline, xsize, ysize, Ltoa_data)

        return Ltoa_data


    def read_block(self, size, offset, bands):

        (ysize, xsize) = size
        nbands = len(bands)

        # initialize block
        block = Block(offset=offset, size=size, bands=bands)

        block.latitude  = self.read_band('latitude',  size, offset)
        block.longitude = self.read_band('longitude', size, offset)

        # read geometry
        block.sza = self.read_band('SZA', size, offset, tiepoint=True)
        block.vza = self.read_band('OZA', size, offset, tiepoint=True)
        block.saa = self.read_band('SAA', size, offset, tiepoint=True)
        block.vaa = self.read_band('OAA', size, offset, tiepoint=True)

        # read TOA
        block.Ltoa = np.zeros((ysize,xsize,nbands)) + np.NaN
        for iband, band in enumerate(bands):
            Ltoa_data = self.read_band(self.band_names[band], size, offset)
            block.Ltoa[:,:,iband] = Ltoa_data[:,:]

        # read F0 for each band
        block.F0 = np.zeros((ysize, xsize, nbands)) + np.NaN
        for iband, band in enumerate(bands):
            F0_data = self.read_band(self.F0_band_names[band], size, offset)
            block.F0[:,:,iband] = F0_data[:,:]

        # read wavelen
        block.wavelen = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            lam0 = self.read_band(self.lambda0_band_names[band], size, offset)
            block.wavelen[:,:,iband] = lam0[:,:]

        warnings.warn('TODO')
        block.jday = 200

        # aux data
        warnings.warn('TODO')
        block.ozone = np.zeros((ysize, xsize)) + 300.
        block.wind_speed = np.zeros((ysize, xsize)) + 5.

        block.bitmask = np.zeros(size, dtype='uint16')

        print 'Read block', block

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


class Level1_OLCI(object):
    '''
    OLCI reader using the netcdf module
    '''
    def __init__(self, dirname, sline=0, eline=-1):

        if not os.path.isdir(dirname):
            dirname = os.path.dirname(dirname)

        self.dirname = dirname
        self.nc_datasets = {}

        # get product shape
        (self.totalheight, self.totalwidth) = self.get_ncroot('Oa01_radiance.nc').variables['Oa01_radiance'].shape
        print 'height={}, width={}'.format(self.totalheight, self.totalwidth)

        self.width = self.totalwidth
        self.height = self.totalheight

        self.blocksize = 200
        self.sline = sline
        self.eline = eline

        if eline < 0:
            self.height = self.totalheight
            self.height -= sline
            self.height += eline + 1
        else:
            self.height = eline-sline

        self.shape = (self.height, self.width)

        # file names
        self.band_names = {
                400 : 'Oa01_radiance', 412 : 'Oa02_radiance',
                443 : 'Oa03_radiance', 490 : 'Oa04_radiance',
                510 : 'Oa05_radiance', 560 : 'Oa06_radiance',
                620 : 'Oa07_radiance', 665 : 'Oa08_radiance',
                674 : 'Oa09_radiance', 681 : 'Oa10_radiance',
                709 : 'Oa11_radiance', 754 : 'Oa12_radiance',
                760 : 'Oa13_radiance', 764 : 'Oa14_radiance',
                767 : 'Oa15_radiance', 779 : 'Oa16_radiance',
                865 : 'Oa17_radiance', 885 : 'Oa18_radiance',
                900 : 'Oa19_radiance', 940 : 'Oa20_radiance',
                1020: 'Oa21_radiance',
                }

        self.band_index = {
                400 : 0, 412: 1, 443 : 2, 490: 3,
                510 : 4, 560: 5, 620 : 6, 665: 7,
                674 : 8, 681: 9, 709 :10, 754: 11,
                760 :12, 764: 13, 767 :14, 779: 15,
                865 :16, 885: 17, 900 :18, 940: 19,
                1020:20}

        self.F0 = self.get_ncroot('instrument_data.nc').variables['solar_flux'][:]
        self.lam0 = self.get_ncroot('instrument_data.nc').variables['lambda0'][:]

    def get_ncroot(self, filename):
        if filename in self.nc_datasets:
            return self.nc_datasets[filename]

        self.nc_datasets[filename] = Dataset(os.path.join(self.dirname, filename))

        return self.nc_datasets[filename]


    def read_band(self, band_name, size, offset):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset

        # load netcdf object if not done already
        if band_name.startswith('Oa'):
            filename = band_name + '.nc'
            tiepoint = False
        elif band_name in ['latitude', 'longitude']:
            filename = 'geo_coordinates.nc'
            tiepoint = False
        elif band_name in ['SZA', 'SAA', 'OZA', 'OAA']:
            filename = 'tie_geometries.nc'
            tiepoint = True
        elif band_name in ['detector_index']:
            filename = 'instrument_data.nc'
            tiepoint = False
        else:
            raise Exception('ERROR')


        root = self.get_ncroot(filename)
        var = root.variables[band_name]

        data = var[yoffset+self.sline:yoffset+self.sline+ysize,
                   xoffset:xoffset+xsize]

        if tiepoint:
            shp = data.shape
            coords = np.meshgrid(np.linspace(0, 76, 1217), np.arange(shp[0]))  # FIXME
            out = np.zeros(size, dtype='float32')
            map_coordinates(data, (coords[1], coords[0]), output=out)
            # FIXME: don't use 3rd order for azimuth angles
            data = out

        return data


    def get_date(self):
        var = self.get_ncroot('Oa01_radiance.nc')
        dstart = datetime.strptime(var.getncattr('start_time'), '%Y-%m-%dT%H:%M:%S.%fZ')

        return dstart

    def read_block(self, size, offset, bands):

        (ysize, xsize) = size
        nbands = len(bands)

        # initialize block
        block = Block(offset=offset, size=size, bands=bands)

        # read geometry
        block.latitude  = self.read_band('latitude',  size, offset)
        block.longitude = self.read_band('longitude', size, offset)

        # read geometry
        block.sza = self.read_band('SZA', size, offset)
        block.vza = self.read_band('OZA', size, offset)
        block.saa = self.read_band('SAA', size, offset)
        block.vaa = self.read_band('OAA', size, offset)

        # read LTOA
        block.Ltoa = np.zeros((ysize,xsize,nbands)) + np.NaN
        for iband, band in enumerate(bands):
            Ltoa_data = self.read_band(self.band_names[band], size, offset)
            block.Ltoa[:,:,iband] = Ltoa_data[:,:]

        # detector index
        di = self.read_band('detector_index', size, offset)

        # solar irradiance
        block.F0 = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.F0[:,:,iband] = self.F0[self.band_index[band], di]

        # detector wavelength
        block.wavelen = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = self.lam0[self.band_index[band], di]

        # julian day
        block.jday = self.get_date().timetuple().tm_yday

        warnings.warn('TODO')
        block.ozone = np.zeros((ysize, xsize)) + 300.
        block.wind_speed = np.zeros((ysize, xsize)) + 5.

        warnings.warn('TODO')
        block.bitmask = np.zeros(size, dtype='uint16')

        print 'Read', block

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





