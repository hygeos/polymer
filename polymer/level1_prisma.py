#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from pathlib import Path
import dateutil
import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import netCDF4
from matplotlib import pyplot as plt
from polymer.ancillary import Ancillary_NASA

from polymer.block import Block
from polymer.common import L2FLAGS
from polymer.level1_nasa import filled
from polymer.utils import coeff_sun_earth_distance, raiseflag
from polymer import prisma



class Level1_PRISMA:
    """
    PRISMA Level1 for Polymer

    http://prisma.asi.it/missionselect/docs/PRISMA%20ATBD_v1.pdf
    http://prisma.asi.it/missionselect/docs/PRISMA%20Product%20Specifications_Is2_3.pdf
    """
    def __init__(self, level1,
                 blocksize=100,
                 ancillary=None,
                 landmask=None,
                 sline=0, eline=-1, scol=0, ecol=-1,
                 ) -> None:
        self.sensor = 'PRISMA'
        level1 = Path(level1)
        self.filename = level1
        self.landmask = landmask
        
        # open level1
        self.h = h5py.File(level1)
        self.HCO = self.h['HDFEOS']['SWATHS']['PRS_L1_HCO']
        self.HCO_DATA = self.HCO['Data Fields']
        self.HCO_GEO = self.HCO['Geolocation Fields']

        # open level2
        level2 = level1.parent/(level1.name.replace('PRS_L1_STD_OFFL_', 'PRS_L2C_STD_'))
        assert level2.exists()
        self.hl2 = h5py.File(level2)
        self.HCO_GEO_L2 = self.hl2['HDFEOS']['SWATHS']['PRS_L2C_HCO']['Geometric Fields']

        self.blocksize = blocksize
        self.sline = sline
        self.scol = scol

        self.totalheight, self.totalwidth = self.HCO_GEO['Latitude_VNIR'].shape

        if ancillary is None:
            self.ancillary = Ancillary_NASA()
        else:
            self.ancillary = ancillary

        if eline < 0:
            self.height = self.totalheight
            self.height -= sline
            self.height += eline + 1
        else:
            self.height = eline-sline

        if ecol < 0:
            self.width = self.totalwidth
            self.width -= scol
            self.width += ecol + 1
        else:
            self.width = ecol - scol

        self.shape = (self.height, self.width)
        print('Initializing PRISMA product of shape', self.shape)

        self.datetime = self.get_time()

        # initialize ancillary data
        self.ozone = self.ancillary.get('ozone', self.datetime)
        self.wind_speed = self.ancillary.get('wind_speed', self.datetime)
        self.surf_press = self.ancillary.get('surf_press', self.datetime)

        self.ancillary_files = OrderedDict()
        self.ancillary_files.update(self.ozone.filename)
        self.ancillary_files.update(self.wind_speed.filename)
        self.ancillary_files.update(self.surf_press.filename)

        self.F0 = prisma.F0

        # landsat data
        if self.landmask is None:
            self.landmask_data = None
        else:
            lat = self.HCO_GEO['Latitude_VNIR'][:,:]
            lon = self.HCO_GEO['Longitude_VNIR'][:,:]
            self.landmask_data = self.landmask.get(lat, lon)


    def attributes(self, datefmt):
        attr = OrderedDict()
        attr['datetime'] = self.datetime
        return attr

    def read_block(self, size, offset, bands):
        nbands = len(bands)
        size3 = size + (nbands,)
        (ysize, xsize) = size
        (yoffset, xoffset) = offset
        SY = slice(offset[0]+self.sline, offset[0]+self.sline+size[0])
        SX = slice(offset[1]+self.scol , offset[1]+self.scol+size[1])

        ibands = np.array([prisma.bands.index(b) for b in bands])

        block = Block(offset=offset, size=size, bands=bands)
        block.jday = self.datetime.timetuple().tm_yday
        block.month = self.datetime.timetuple().tm_mon

        block.latitude = self.HCO_GEO['Latitude_VNIR'][SY, SX]
        block.longitude = self.HCO_GEO['Longitude_VNIR'][SY, SX]
        block.sza = self.HCO_GEO_L2['Solar_Zenith_Angle'][SY, SX]
        block.vza = self.HCO_GEO_L2['Observing_Angle'][SY, SX]
        block._raa = self.HCO_GEO_L2['Rel_Azimuth_Angle'][SY, SX]

        # read radiometry
        scale_vnir = self.h.attrs['ScaleFactor_Vnir']
        offset_vnir = self.h.attrs['Offset_Vnir']
        mask_value = 65535
        Ltoa_VNIR_raw = self.HCO_DATA['VNIR_Cube'][
            SY,:,SX].transpose([0, 2, 1])[:,:,len(prisma.bands)-ibands-1]
        Ltoa_VNIR = offset_vnir + Ltoa_VNIR_raw/scale_vnir
        Ltoa_VNIR[Ltoa_VNIR>=mask_value] = np.nan
        block.Ltoa = Ltoa_VNIR/10.  # convert W/m^2/um/sr -> mW/cm^2/um/sr
        doy = self.datetime.timetuple().tm_yday  # day of year [1-366]
        block.F0 = np.array([self.F0[b] for b in bands])*coeff_sun_earth_distance(doy)
        block.F0 = np.broadcast_to(block.F0, size3)
        block.cwavelen = np.array(prisma.wav, dtype='float32')[ibands]
        block.wavelen = np.broadcast_to(
            block.cwavelen,
            size3).copy()

        # Initialize bitmask
        block.bitmask = np.zeros(size, dtype='uint16')

        # Stripes in there
        if self.landmask is None:
            raiseflag(
                block.bitmask, L2FLAGS['LAND'],
                self.HCO_DATA['LandCover_Mask'][
                    yoffset+self.sline:yoffset+self.sline+ysize,
                    xoffset+self.scol:xoffset+self.scol+xsize,
                ])
        else:
            raiseflag(
                block.bitmask, L2FLAGS['LAND'],
                self.landmask_data[
                    yoffset+self.sline:yoffset+self.sline+ysize,
                    xoffset+self.scol:xoffset+self.scol+xsize,
                ])

        # ancillary data
        block.ozone = np.zeros(size, dtype='float32')
        block.ozone[:] = self.ozone[block.latitude, block.longitude]
        block.wind_speed = np.zeros(size, dtype='float32')
        block.wind_speed[:] = self.wind_speed[block.latitude, block.longitude]
        block.surf_press = np.zeros(size, dtype='float32')
        block.surf_press[:] = self.surf_press[block.latitude, block.longitude]

        block.altitude = np.zeros(size, dtype='float32')

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


    def get_time(self):
        isotime = self.h.attrs['Product_StartTime']
        time = dateutil.parser.parse(isotime)
        return time

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
