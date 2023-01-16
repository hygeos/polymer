#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import numpy as np
from polymer.block import Block
from polymer.common import L2FLAGS
from polymer.utils import raiseflag
from polymer.level1 import Level1_base
from polymer.bodhaine import rod
from polymer.luts import read_mlut, LUT
from netCDF4 import Dataset
from scipy.ndimage import map_coordinates
from datetime import datetime
from polymer.ancillary import Ancillary_NASA
import os
from collections import OrderedDict



class Level1_SAFE(Level1_base):
    '''
    OLCI reader using the netcdf module

    landmask:
        * None => don't apply land mask at all
        * 'default' => use landmask provided in Level1
        * GSW object: use global surface water product (see gsw.py)

    altitude: surface altitude in m
        * a float
        * a DEM instance such as:
            SRTM3(cache_dir=...)  # srtm.py
            GLOBE(directory=...)  # globe.py
            SRTM3(..., missing=GLOBE(...))
    '''
    def __init__(self, dirname,
                 sline=0, eline=-1,
                 scol=0, ecol=-1,
                 blocksize=100, ancillary=None,
                 landmask='default',
                 altitude=0.,
                 sensor=None,
                 central_wavelength=None,
                 band_names=None,
                 band_index=None,
                 Ltyp=None,
                 sigma_typ=None,
                 add_noise=False,
                 ):

        self.sensor = sensor
        self.blocksize = blocksize
        self.landmask = landmask
        self.altitude = altitude

        self.central_wavelength = central_wavelength
        self.band_names = band_names
        self.band_index = band_index
        self.sigma_typ = sigma_typ
        self.Ltyp = Ltyp
        self.add_noise = add_noise

        if not os.path.isdir(dirname):
            dirname = os.path.dirname(dirname)

        if dirname.endswith(os.path.sep):
            dirname = dirname[:-1]

        self.dirname = dirname
        self.filename = dirname
        self.ancillary = ancillary
        self.nc_datasets = {}

        # get product shape
        bid = list(self.band_names.values())[0]
        (totalheight, totalwidth) = self.get_ncroot(bid+'.nc').variables[bid].shape
        print('height={}, width={}'.format(totalheight, totalwidth))

        self.init_shape(
                totalheight=totalheight,
                totalwidth=totalwidth,
                sline=sline,
                eline=eline,
                scol=scol,
                ecol=ecol)

        self.F0 = self.get_ncroot('instrument_data.nc').variables['solar_flux'][:]
        self.lam0 = self.get_ncroot('instrument_data.nc').variables['lambda0'][:]

        # read quality flag meanings
        qf = self.get_ncroot('qualityFlags.nc').variables['quality_flags']
        fmask = qf.getncattr('flag_masks')
        fmeaning = str(qf.getncattr('flag_meanings')).split()
        self.quality_flags = {}
        for i in range(len(fmeaning)):
            self.quality_flags[fmeaning[i]] = fmask[i]

        # date initialization
        self.read_date()

        # ancillary data initialization
        self.ancillary_files = OrderedDict()
        if self.ancillary is not None:
            self.init_ancillary()

        self.init_landmask()

    def read_date(self):
        bid = list(self.band_names.values())[0]
        var = self.get_ncroot(bid+'.nc')
        self.dstart = datetime.strptime(var.getncattr('start_time'), '%Y-%m-%dT%H:%M:%S.%fZ')
        self.dstop  = datetime.strptime(var.getncattr('stop_time'), '%Y-%m-%dT%H:%M:%S.%fZ')


    def init_ancillary(self):
        self.ozone = self.ancillary.get('ozone', self.date())
        self.wind_speed = self.ancillary.get('wind_speed', self.date())
        self.surf_press = self.ancillary.get('surf_press', self.date())
        self.ancillary_files.update(self.ozone.filename)
        self.ancillary_files.update(self.wind_speed.filename)
        self.ancillary_files.update(self.surf_press.filename)

        self.ancillary_files.update(self.wind_speed.filename)

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

    def get_ncroot(self, filename):
        if filename in self.nc_datasets:
            return self.nc_datasets[filename]

        self.nc_datasets[filename] = Dataset(os.path.join(self.dirname, filename))
        self.nc_datasets[filename].set_auto_mask(False)

        return self.nc_datasets[filename]


    def read_band(self, band_name, size, offset):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset

        # load netcdf object if not done already
        if band_name in ['latitude', 'longitude']:
            filename = 'geo_coordinates.nc'
            tiepoint = False
        elif band_name in ['SZA', 'SAA', 'OZA', 'OAA']:
            filename = 'tie_geometries.nc'
            tiepoint = True
        elif band_name in ['detector_index']:
            filename = 'instrument_data.nc'
            tiepoint = False
        elif band_name in ['total_ozone', 'sea_level_pressure', 'horizontal_wind']:
            filename = 'tie_meteo.nc'
            tiepoint = True
        elif band_name in ['quality_flags']:
            filename = 'qualityFlags.nc'
            tiepoint = False
        else:
            filename = band_name + '.nc'
            tiepoint = False


        root = self.get_ncroot(filename)
        try:
            var = root.variables[band_name]
        except KeyError:
            print(root.variables)
            raise

        if tiepoint:
            # window for tiepoint data read
            ac = root.getncattr('ac_subsampling_factor')
            al = root.getncattr('al_subsampling_factor')

            ymin = int(np.floor((yoffset+self.sline)/al))
            ymax = min(int(np.ceil((yoffset+self.sline+ysize-1)/al)), var.shape[0])
            xmin = int(np.floor((xoffset+self.scol)/ac))
            xmax = min(int(np.ceil((xoffset+self.scol+xsize-1)/ac)), var.shape[1])
            data = var[ymin:ymax+1, xmin:xmax+1]

            if band_name == 'horizontal_wind':
                # wind modulus from zonal and meridional components
                data = np.sqrt(data[...,0]**2 + data[...,1]**2)

            coords = np.meshgrid(np.linspace(0, xmax-xmin, (xmax-xmin)*ac+1),
                                 np.linspace(0, ymax-ymin, (ymax-ymin)*al+1))
            out = np.zeros(coords[0].shape, dtype='float32')
            if band_name in ['OAA', 'SAA']:
                order=0   # nearest neighbour for azimuth angles
            else:
                order=1
            map_coordinates(data, (coords[1], coords[0]), output=out, order=order)
            data = out

            data = data[
                yoffset+self.sline-al*ymin : yoffset+self.sline-al*ymin+ysize,
                xoffset+self.scol-ac*xmin : xoffset+self.scol-ac*xmin+xsize,
            ]

        else:
            data = var[
                yoffset+self.sline:yoffset+self.sline+ysize,
                xoffset+self.scol:xoffset+self.scol+xsize,
                ]

        return data


    def date(self):
        return self.dstart + (self.dstop - self.dstart)//2


    def read_block(self, size, offset, bands):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset
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
            if self.add_noise:
                stdev = np.sqrt(Ltoa_data/self.Ltyp[band])*self.sigma_typ[band]
                noise = stdev*np.random.normal(0, 1, stdev.size).reshape(stdev.shape)
            else:
                noise = 0
            block.Ltoa[:,:,iband] = Ltoa_data[:,:] + noise

        # detector index
        di = self.read_band('detector_index', size, offset)

        # solar irradiance (seasonally corrected)
        block.F0 = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.F0[:,:,iband] = self.F0[self.band_index[band], di]

        # detector wavelength
        block.wavelen = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        block.cwavelen = np.zeros(nbands, dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = self.lam0[self.band_index[band], di]
            block.cwavelen[iband] = self.central_wavelength[band]

        # julian day and month
        block.jday = self.date().timetuple().tm_yday
        block.month = self.date().timetuple().tm_mon

        # quality flags
        bitmask = self.read_band('quality_flags', size, offset)
        block.bitmask = np.zeros(size, dtype='uint16')
        if self.landmask == 'default':
            # raise LAND mask when land is raised but not fresh_inland_water
            bval = self.quality_flags['land'] + self.quality_flags['fresh_inland_water']
            raiseflag(block.bitmask, L2FLAGS['LAND'],
                      bitmask & bval == self.quality_flags['land'])
        elif self.landmask is None:
            pass
        else:  # assume GSW-like object
            raiseflag(block.bitmask, L2FLAGS['LAND'],
                      self.landmask_data[
                          yoffset:yoffset+ysize,
                          xoffset:xoffset+xsize,
                                         ])

        l1_invalid = bitmask & self.quality_flags['invalid'] != 0
        raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], l1_invalid)

        # read ancillary data
        if self.ancillary is not None:
            # external ancillary files
            block.ozone = np.zeros_like(block.latitude, dtype=self.ozone.dtype) + np.NaN
            block.ozone[~l1_invalid] = self.ozone[
                block.latitude[~l1_invalid],
                block.longitude[~l1_invalid]]
            block.wind_speed = np.zeros_like(block.latitude, dtype=self.wind_speed.dtype) + np.NaN
            block.wind_speed[~l1_invalid] = self.wind_speed[
                block.latitude[~l1_invalid],
                block.longitude[~l1_invalid]]
            P0 = np.zeros_like(block.latitude, dtype=self.surf_press.dtype) + np.NaN
            P0[~l1_invalid] = self.surf_press[
                block.latitude[~l1_invalid],
                block.longitude[~l1_invalid]]

        else: # ancillary files embedded in level1

            # read total ozone in kg/m2
            block.ozone = self.read_band('total_ozone', size, offset)
            block.ozone /= 2.1415e-5  # convert kg/m2 to DU

            # read sea level pressure in hPa
            P0 = self.read_band('sea_level_pressure', size, offset)

            # read wind speed
            block.wind_speed = self.read_band('horizontal_wind', size, offset)

        # read surface altitude
        try:
            block.altitude = self.altitude.get(lat=block.latitude,
                                               lon=block.longitude)
        except AttributeError:
            # altitude expected to be a float
            block.altitude = np.zeros((ysize, xsize), dtype='float32') + self.altitude

        # calculate surface altitude
        block.surf_press = P0 * np.exp(-block.altitude/8000.)

        if self.Ltyp is not None:
            block.Ltyp = np.array([self.Ltyp[b] for b in bands], dtype='float32')
        if self.sigma_typ is not None:
            block.sigma_typ = np.array([self.sigma_typ[b] for b in bands],
                                       dtype='float32')

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
        attr = OrderedDict()
        attr['l1_filename'] = self.filename
        attr['start_time'] = self.dstart.strftime(datefmt)
        attr['stop_time'] = self.dstop.strftime(datefmt)
        attr['central_wavelength'] = str(self.central_wavelength)

        attr.update(self.ancillary_files)

        return attr

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

