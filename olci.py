#!/usr/bin/env python
# -*- coding: utf-8 -*-

from params import Params
import numpy as np
from block import Block
from common import L2FLAGS
from netCDF4 import Dataset
from scipy.ndimage import map_coordinates
from datetime import datetime
import os


class Params_OLCI(Params):

    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()

        # FIXME
        self.bands_corr = [    412,443,490,510,560,620,665,754,779,865]
        self.bands_oc   = [    412,443,490,510,560,620,665,754,779,865]
        self.bands_rw   = [400,412,443,490,510,560,620,665,754,779,865]

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

        # from SeaDAS v7.3.2
        self.K_OZ = {
                400 : 2.985E-06, 412 : 2.341E-04,
                443 : 2.897E-03, 490 : 2.066E-02,
                510 : 4.129E-02, 560 : 1.058E-01,
                620 : 1.085E-01, 665 : 5.005E-02,
                674 : 4.095E-02, 681 : 3.507E-02,
                709 : 1.887E-02, 754 : 8.743E-03,
                760 : 6.713E-03, 764 : 6.916E-03,
                768 : 6.754E-03, 779 : 7.700E-03,
                865 : 2.156E-03, 885 : 1.226E-03,
                900 : 1.513E-03, 940 : 7.120E-04,
                1020: 8.448E-05,
                }

        # from SeaDAS v7.3.2
        self.K_NO2 = {
                400 : 6.175E-19, 412 : 6.083E-19,
                443 : 4.907E-19, 490 : 2.933E-19,
                510 : 2.187E-19, 560 : 7.363E-20,
                620 : 2.818E-20, 665 : 6.645E-21,
                674 : 1.014E-20, 681 : 6.313E-21,
                709 : 4.938E-21, 754 : 1.379E-21,
                761 : 4.472E-22, 764 : 6.270E-22,
                768 : 5.325E-22, 779 : 3.691E-22,
                865 : 2.868E-23, 885 : 4.617E-23,
                900 : 5.512E-23, 940 : 3.167E-24,
                1020: 0.000E+00,
                }

        self.update(**kwargs)



class Level1_OLCI(object):
    '''
    OLCI reader using the netcdf module
    '''
    def __init__(self, dirname, sline=0, eline=-1, blocksize=100):

        if not os.path.isdir(dirname):
            dirname = os.path.dirname(dirname)

        self.dirname = dirname
        self.filename = dirname
        self.nc_datasets = {}

        # get product shape
        (self.totalheight, self.totalwidth) = self.get_ncroot('Oa01_radiance.nc').variables['Oa01_radiance'].shape
        print 'height={}, width={}'.format(self.totalheight, self.totalwidth)

        self.width = self.totalwidth
        self.height = self.totalheight

        self.blocksize = blocksize
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

        # read quality flag meanings
        qf = self.get_ncroot('qualityFlags.nc').variables['quality_flags']
        fmask = qf.getncattr('flag_masks')
        fmeaning = str(qf.getncattr('flag_meanings')).split()
        self.quality_flags = {}
        for i in xrange(len(fmeaning)):
            self.quality_flags[fmeaning[i]] = fmask[i]


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
        elif band_name in ['total_ozone', 'sea_level_pressure']:
            filename = 'tie_meteo.nc'
            tiepoint = True
        elif band_name in ['quality_flags']:
            filename = 'qualityFlags.nc'
            tiepoint = False
        else:
            raise Exception('ERROR')


        root = self.get_ncroot(filename)
        var = root.variables[band_name]

        data = var[yoffset+self.sline:yoffset+self.sline+ysize,
                   xoffset:xoffset+xsize]

        if tiepoint:
            shp = data.shape
            coords = np.meshgrid(np.linspace(0, shp[1]-1, self.totalwidth), np.arange(shp[0]))
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

        # read total ozone in kg/m2
        block.ozone = self.read_band('total_ozone', size, offset)
        block.ozone /= 2.1415e-5  # convert kg/m2 to DU

        # read sea level pressure in hPa
        block.surf_press = self.read_band('sea_level_pressure', size, offset)

        block.wind_speed = np.zeros((ysize, xsize)) + 5.   # FIXME

        # quality flags
        bitmask = self.read_band('quality_flags', size, offset)
        block.bitmask = np.zeros(size, dtype='uint16')
        block.bitmask += L2FLAGS['LAND']*(bitmask & self.quality_flags['land'] != 0).astype('uint16')


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





