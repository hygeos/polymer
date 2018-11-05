#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from polymer.level1 import Level1_base
from polymer.block import Block
from netCDF4 import Dataset
import numpy as np
from warnings import warn
from datetime import datetime
from polymer.common import L2FLAGS
from polymer.utils import raiseflag, coeff_sun_earth_distance
from polymer.ancillary import Ancillary_NASA
from polymer.level1_meris import central_wavelength_meris
from polymer.level1_olci import central_wavelength_olci
from os.path import basename, join, dirname
from collections import OrderedDict
import pandas as pd



class Level1_NETCDF(Level1_base):
    '''
    Reader for subsetted products in netcdf format
    (produced by SNAP)

    Supported sensors:
        * OLCI
        * MERIS
        * Sentinel2

    landmask:
        * None => don't apply land mask at all
        * 'default' => use landmask provided in Level1
        * GSW object: use global surface water product (see gsw.py)

    altitude: surface altitude in m
        * a float
        * a DEM instance such as:
            SRTM(cache_dir=...)  # srtm.py
            GLOBE(directory=...)  # globe.py
            SRTM(..., missing=GLOBE(...))
    '''
    def __init__(self, filename,
                 blocksize=(500, 400),
                 dir_smile=None,
                 apply_land_mask=None,
                 landmask=None,
                 altitude=0.,
                 ancillary=None):

        self.filename = filename
        self.root = Dataset(filename)
        self.blocksize = blocksize
        self.landmask = landmask
        self.altitude = altitude
        if apply_land_mask is not None:
            raise Exception('[Deprecated] Level1_OLCI: apply_land_mask option has been replaced by landmask.')

        # detect sensor
        try:
            title = self.root.getncattr('title')
        except AttributeError:
            title = self.root.getncattr('product_type')

        if 'MERIS' in title:
            self.sensor = 'MERIS'
            self.varnames = {
                    'latitude': 'latitude',
                    'longitude': 'longitude',
                    'SZA': 'sun_zenith',
                    'VZA': 'view_zenith',
                    'SAA': 'sun_azimuth',
                    'VAA': 'view_azimuth',
                    'ozone': 'ozone',
                    'zwind': 'zonal_wind',
                    'mwind': 'merid_wind',
                    'press': 'atm_press',
                    }
            BANDS_MERIS = [412, 443, 490, 510, 560,
                           620, 665, 681, 709, 754,
                           760, 779, 865, 885, 900]
            self.band_index = dict((b, i+1) for (i, b) in enumerate(BANDS_MERIS))

            # read detector wavelength and solar irradiance
            if dir_smile is None:
                dir_smile = join(dirname(dirname(__file__)), 'auxdata/meris/smile/v2/')

            if 'MERIS Full Resolution' in title:
                self.F0 = np.genfromtxt(join(dir_smile, 'sun_spectral_flux_fr.txt'), names=True)
                self.detector_wavelength = np.genfromtxt(join(dir_smile, 'central_wavelen_fr.txt'), names=True)
            elif 'MERIS Reduced Resolution' in title:
                self.F0 = np.genfromtxt(join(dir_smile, 'sun_spectral_flux_rr.txt'), names=True)
                self.detector_wavelength = np.genfromtxt(join(dir_smile, 'central_wavelen_rr.txt'), names=True)
            else:
                raise Exception('Invalid "{}"'.format(title))

        elif 'OLCI' in title:
            self.sensor = 'OLCI'
            self.varnames = {
                    'latitude': 'latitude',
                    'longitude': 'longitude',
                    'SZA': 'SZA',
                    'VZA': 'OZA',
                    'SAA': 'SAA',
                    'VAA': 'OAA',
                    'ozone': 'total_ozone',
                    'zwind': 'horizontal_wind_vector_1',
                    'mwind': 'horizontal_wind_vector_2',
                    'press': 'sea_level_pressure',
                    }
            BANDS_OLCI = [
                    400 , 412, 443 , 490, 510 , 560,
                    620 , 665, 674 , 681, 709 , 754,
                    760 , 764, 767 , 779, 865 , 885,
                    900 , 940, 1020]
            self.band_index = dict((b, i+1) for (i, b) in enumerate(BANDS_OLCI))

        elif title == 'S2_MSI_Level-1C':
            self.sensor = 'MSI'
            self.varnames = {
                    'latitude': 'lat',
                    'longitude': 'lon',
                    'SZA': 'sun_zenith',
                    'VZA': 'view_zenith_B1',
                    'SAA': 'sun_azimuth',
                    'VAA': 'view_azimuth_B1',
                    }

            # get platform name
            metadata = self.root.variables['metadata']
            self.platform = metadata.getncattr('Level-1C_User_Product:General_Info:Product_Info:Datatake:SPACECRAFT_NAME')
            self.platform = self.platform.replace('entinel-', '')
        else:
            raise Exception('Could not identify sensor from "{}"'.format(title))

        # get product shape
        totalheight = self.root.variables[self.varnames['latitude']].shape[0]
        totalwidth = self.root.variables[self.varnames['latitude']].shape[1]

        print('{} product, size is {}x{}'.format(self.sensor, totalheight, totalwidth))

        self.init_shape(
                totalheight=totalheight,
                totalwidth=totalwidth,
                sline=0, eline=-1,
                scol=0,  ecol=-1)

        self.init_date()

        # init ancillary
        if (ancillary is None) and (self.sensor == 'MSI'):
            self.ancillary = Ancillary_NASA()
        else:
            self.ancillary = ancillary
        if self.ancillary is not None:
            self.init_ancillary()

        self.init_bands()

        self.init_landmask()


    def init_bands(self):
        if self.sensor == 'MSI':
            band_names = {
                    443 : 'B01', 490 : 'B02',
                    560 : 'B03', 665 : 'B04',
                    705 : 'B05', 740 : 'B06',
                    783 : 'B07', 842 : 'B08',
                    865 : 'B8A', 945 : 'B09',
                    1375: 'B10', 1610: 'B11',
                    2190: 'B12',
                    }
            dir_aux_msi = join(dirname(dirname(__file__)), 'auxdata', 'msi')
            srf_file = join(dir_aux_msi, 'S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_{}.csv'.format(self.platform))

            srf_data = pd.read_csv(srf_file)

            wav = srf_data.SR_WL

            self.wav = OrderedDict()
            for b, bn in band_names.items():
                col = self.platform + '_SR_AV_' + bn.replace('B0', 'B')
                srf = srf_data[col]
                wav_eq = np.trapz(wav*srf)/np.trapz(srf)
                self.wav[b] = wav_eq

        elif self.sensor == 'OLCI':
            self.central_wavelength = central_wavelength_olci
        elif self.sensor == 'MERIS':
            self.central_wavelength = central_wavelength_meris
        else:
            raise Exception('Invalid sensor "{}"'.format(self.sensor))

    def init_landmask(self):
        if not hasattr(self.landmask, 'get'):
            return

        lat = self.read_band('latitude',
                             (self.height, self.width),
                             (0, 0))
        lon = self.read_band('longitude',
                             (self.height, self.width),
                             (0, 0))

        self.landmask_data = self.landmask.get(lat, lon)


    def init_ancillary(self):
        self.ozone = self.ancillary.get('ozone', self.date)
        self.wind_speed = self.ancillary.get('wind_speed', self.date)
        self.surf_press = self.ancillary.get('surf_press', self.date)

        self.ancillary_files = OrderedDict()
        self.ancillary_files.update(self.ozone.filename)
        self.ancillary_files.update(self.wind_speed.filename)
        self.ancillary_files.update(self.surf_press.filename)

    def read_date(self, date):
        ''' parse a date in the format 04-JUL-2017 12:31:28.013924 '''
        date = date.replace('JAN', '01')
        date = date.replace('FEB', '02')
        date = date.replace('MAR', '03')
        date = date.replace('APR', '04')
        date = date.replace('MAY', '05')
        date = date.replace('JUN', '06')
        date = date.replace('JUL', '07')
        date = date.replace('AUG', '08')
        date = date.replace('SEP', '09')
        date = date.replace('OCT', '10')
        date = date.replace('NOV', '11')
        date = date.replace('DEC', '12')

        # remove the milliseconds
        date = date[:date.index('.')]

        return datetime.strptime(date, '%d-%m-%Y %H:%M:%S')


    def init_date(self):

        date1 = self.root.getncattr('start_date')
        date2 = self.root.getncattr('stop_date')

        date1 = self.read_date(date1)
        date2 = self.read_date(date2)

        self.dstart = date1
        self.dstop  = date2

        self.date = date1 + (date2 - date1)//2

    def get_bitmask(self, flags_dataset, flag_name, size, offset):
        flags = self.read_band(flags_dataset, size, offset)
        flag_meanings = self.root.variables[flags_dataset].getncattr('flag_meanings').split()
        flag_masks = self.root.variables[flags_dataset].getncattr('flag_masks')
        masks = dict(zip(flag_meanings, flag_masks))

        return flags & masks[flag_name] != 0


    def read_band(self, band_name, size, offset):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset

        if band_name not in self.root.variables:
            for x in [str(x) for x in self.root.variables]:
                print('->', x)
            raise Exception('{} not in dataset'.format(band_name))

        var = self.root.variables[band_name]

        data = var[yoffset+self.sline:yoffset+self.sline+ysize,
                   xoffset+self.scol :xoffset+self.scol+xsize,
                  ]

        if 'ozone' in band_name:
            if 'units' in var.ncattrs():
                ozone_unit = var.getncattr('units')
            else:
                ozone_unit = {
                        'MERIS': 'DU',
                        'OLCI': 'Kg.m-2',
                        }[self.sensor]

            if ozone_unit == 'Kg.m-2':
                data /= 2.1415e-5  # convert kg/m2 to DU
            elif ozone_unit == 'DU':
                pass
            else:
                raise Exception('Error in ozone unit ({})'.format(ozone_unit))

        return np.array(data)

    def read_block(self, size, offset, bands):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset
        nbands = len(bands)
        size3 = size + (nbands,)

        # initialize block
        block = Block(offset=offset, size=size, bands=bands)

        # read lat/lon
        block.latitude  = self.read_band('lat', size, offset)
        block.longitude = self.read_band('lon', size, offset)


        # read geometry
        block.sza = self.read_band(self.varnames['SZA'], size, offset)
        block.vza = self.read_band(self.varnames['VZA'], size, offset)
        block.saa = self.read_band(self.varnames['SAA'], size, offset)
        block.vaa = self.read_band(self.varnames['VAA'], size, offset)

        # read Rtoa or Ltoa+F0
        # and wavelen
        block.wavelen = np.zeros(size3, dtype='float32') + np.NaN
        block.cwavelen = np.zeros(nbands, dtype='float32') + np.NaN
        if self.sensor == 'MSI':
            # read Rtoa
            block.Rtoa = np.zeros(size3) + np.NaN
            for iband, band in enumerate(bands):
                band_name = {
                        443 : 'B1', 490 : 'B2',
                        560 : 'B3', 665 : 'B4',
                        705 : 'B5', 740 : 'B6',
                        783 : 'B7', 842 : 'B8',
                        865 : 'B8A', 945 : 'B9',
                        1375: 'B10', 1610: 'B11',
                        2190: 'B12'}[band]

                block.Rtoa[:,:,iband] = self.read_band(band_name, size, offset)

            # init wavelengths
            for iband, band in enumerate(bands):
                block.wavelen[:,:,iband] = self.wav[band]
                block.cwavelen[iband] = self.wav[band]

        elif self.sensor in ['MERIS', 'OLCI']:
            # read Ltoa and F0
            block.Ltoa = np.zeros(size3) + np.NaN
            for iband, band in enumerate(bands):
                if self.sensor == 'MERIS':
                    band_name = 'radiance_{}'.format(self.band_index[band])
                elif self.sensor == 'OLCI':
                    band_name = 'Oa{:02d}_radiance'.format(self.band_index[band])
                else:
                    raise Exception('Invalid sensor "{}"'.format(self.sensor))

                block.Ltoa[:,:,iband] = self.read_band(band_name, size, offset)

            # detector wavelength and solar irradiance
            block.F0 = np.zeros(size3, dtype='float32') + np.NaN
            if self.sensor == 'MERIS':
                # read detector index
                detector_index = self.read_band('detector_index', size, offset)

                for iband, band in enumerate(bands):
                    name = 'lam_band{}'.format(self.band_index[band]-1)   # 0-based
                    block.wavelen[:,:,iband] = self.detector_wavelength[name][detector_index]
                    block.cwavelen[iband] = central_wavelength_meris[band]

                    name = 'E0_band{}'.format(self.band_index[band]-1)   # 0-based
                    block.F0[:,:,iband] = self.F0[name][detector_index]

                coef = coeff_sun_earth_distance(self.date.timetuple().tm_yday)
                block.F0 *= coef

            elif self.sensor == 'OLCI':  # OLCI
                for iband, band in enumerate(bands):
                    block.wavelen[:,:,iband] = self.read_band('lambda0_band_{}'.format(self.band_index[band]), size, offset)
                    block.cwavelen[iband] = central_wavelength_olci[band]
                    block.F0[:,:,iband] = self.read_band('solar_flux_band_{}'.format(self.band_index[band]), size, offset)  # should be seasonally corrected
            else:
                raise Exception('Invalid sensor "{}"'.format(self.sensor))
        else:
            raise Exception('Invalid sensor "{}"'.format(self.sensor))

        # read bitmask
        block.bitmask = np.zeros(size, dtype='uint16')
        if self.sensor == 'OLCI':
            if self.landmask == 'default':
                raiseflag(block.bitmask, L2FLAGS['LAND'], self.get_bitmask('quality_flags', 'land', size, offset))
            raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], self.get_bitmask('quality_flags', 'invalid', size, offset))
            raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], self.get_bitmask('quality_flags', 'cosmetic', size, offset))

        elif self.sensor == 'MERIS':
            if self.landmask == 'default':
                raiseflag(block.bitmask, L2FLAGS['LAND'], self.get_bitmask('l1_flags', 'LAND_OCEAN', size, offset))
            raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], self.get_bitmask('l1_flags', 'INVALID', size, offset))
            raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], self.get_bitmask('l1_flags', 'SUSPECT', size, offset))
            raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], self.get_bitmask('l1_flags', 'COSMETIC', size, offset))
        elif self.sensor == 'MSI':
            raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], np.isnan(block.vza))
        else:
            raise Exception('Invalid sensor {}'.format(self.sensor))
        
        # GSW landmask
        if hasattr(self.landmask, 'get'): # assume GSW-like object
            raiseflag(block.bitmask, L2FLAGS['LAND'],
                      self.landmask_data[
                          yoffset:yoffset+ysize,
                          xoffset:xoffset+xsize,
                                         ])



        # date
        block.jday = self.date.timetuple().tm_yday
        block.month = self.date.timetuple().tm_mon

        # read ancillary data
        if self.sensor in ['MERIS', 'OLCI']:
            block.ozone = self.read_band(self.varnames['ozone'], size, offset)
            zwind = self.read_band(self.varnames['zwind'], size, offset)
            mwind = self.read_band(self.varnames['mwind'], size, offset)
            block.wind_speed = np.sqrt(zwind**2 + mwind**2)
            P0 = self.read_band(self.varnames['press'], size, offset)
        elif self.sensor == 'MSI':
            block.ozone = self.ozone[block.latitude, block.longitude]
            block.wind_speed = self.wind_speed[block.latitude, block.longitude]
            P0 = self.surf_press[block.latitude, block.longitude]
        else:
            raise Exception('Invalid sensor {}'.format(self.sensor))

        # read altitude
        try:
            block.altitude = self.altitude.get(lat=block.latitude,
                                               lon=block.longitude)
        except AttributeError:
            # altitude expected to be a float
            block.altitude = np.zeros((ysize, xsize), dtype='float32') + self.altitude

        # surface pressure
        block.surf_press = P0 * np.exp(-block.altitude/8000.)

        return block

    def attributes(self, datefmt):
        attr = OrderedDict()
        attr['l1_filename'] = self.filename
        attr['start_time'] = self.dstart.strftime(datefmt)
        attr['stop_time'] = self.dstop.strftime(datefmt)

        return attr

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


