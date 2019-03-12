#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from polymer.level1 import Level1_base
from polymer.block import Block
from polymer.hico import bands_hico, wav_hico, F0_hico
import numpy as np
from warnings import warn
#import h5py
from netCDF4 import Dataset
from datetime import datetime
from polymer.ancillary import Ancillary_NASA
from polymer.utils import raiseflag
from polymer.common import L2FLAGS
#from numpy import interp
from os.path import basename, join, dirname
from collections import OrderedDict

class Level1_HICO(object):
    """
    HICO Level1 reader

    landmask:
        * None: no land mask [default]
        * A GSW instance (see gsw.py)
            Example: landmask=GSW(directory='/path/to/gsw_data/')
    """
    def __init__(self, filename, blocksize=200,
                 sline=0, eline=-1, scol=0, ecol=-1,
                 ancillary=None, landmask=None):
        self.nc=Dataset(filename)         #self.h5 = h5py.File(filename) #original h5 code
        self.sensor = 'HICO'
        self.filename = filename
        self.landmask = landmask
        
        self.Lt = self.nc.groups['products']['Lt']  #self.Lt = self.h5['products']['Lt']  #original h5 code
        #print(self.Lt) 

        self.totalheight, self.totalwidth, nlam = self.Lt.shape
        #print(self.totalheight) 
        #print(self.totalwidth) 
        #print(self.Lt.shape) 

        self.blocksize = blocksize
        self.sline = sline
        self.scol = scol

        if ancillary is None:
            self.ancillary = Ancillary_NASA()
        else:
            self.ancillary = ancillary
        #print(self.ancillary)
        
        if eline < 0:
            self.height = self.totalheight
            self.height -= sline
            self.height += eline + 1
        else:
            self.height = eline-sline
        #print(eline)
        
        if ecol < 0:
            self.width = self.totalwidth
            self.width -= scol
            self.width += ecol + 1
        else:
            self.width = ecol - scol
        #print(ecol)
        
        self.shape = (self.height, self.width)
        print('Initializing HICO product of size', self.shape)

        self.datetime = self.get_time()
        #print(self.datetime)

        # initialize ancillary data
        self.ozone = self.ancillary.get('ozone', self.datetime)
        #print(self.ozone,"is of type", type(self.ozone))
        self.wind_speed = self.ancillary.get('wind_speed', self.datetime)
        #print(self.wind_speed,"is of type", type(self.wind_speed))
        self.surf_press = self.ancillary.get('surf_press', self.datetime)
        #print(self.surf_press,"is of type", type(self.surf_press))

        self.ancillary_files = OrderedDict()
        self.ancillary_files.update(self.ozone.filename)
        #print(self.ozone.filename)
        self.ancillary_files.update(self.wind_speed.filename)
        #print(self.wind_speed.filename)
        self.ancillary_files.update(self.surf_press.filename)
        #print(self.surf_press.filename)
        self.init_landmask()
        #print(self.init_landmask)
        #import sys
        #sys.exit(1)


    def init_landmask(self):
        if not hasattr(self.landmask, 'get'):
            return
        #lat = self.h5['navigation']['latitudes'][:,:] #original h5 code
        #lon = self.h5['navigation']['longitudes'][:,:] #original h5 code
        
        lat=self.nc.groups['navigation']['latitudes'][:,:]
        lon=self.nc.groups['navigation']['longitudes'][:,:]

        self.landmask_data = self.landmask.get(lat, lon)
    
    
    def get_time(self):
        #beg_date = self.h5['metadata']['FGDC']['Identification_Information']['Time_Period_of_Content'].attrs['Beginning_Date'].decode('ascii') #original h5 code
        #beg_time = self.h5['metadata']['FGDC']['Identification_Information']['Time_Period_of_Content'].attrs['Beginning_Time'].decode('ascii') #original h5 code
        metadata=self.nc.groups['metadata']['FGDC']['Identification_Information']['Time_Period_of_Content']
        beg_date = metadata.getncattr('Beginning_Date')
        beg_time = metadata.getncattr('Beginning_Time')
        return datetime.strptime(beg_date + beg_time, '%Y%m%d%H%M%S')


    def read_block(self, size, offset, bands):
        nbands = len(bands)
        size3 = size + (nbands,)
        (ysize, xsize) = size
        (yoffset, xoffset) = offset
        SY = slice(offset[0]+self.sline, offset[0]+self.sline+size[0])
        SX = slice(offset[1]+self.scol , offset[1]+self.scol+size[1])

        block = Block(offset=offset, size=size, bands=bands)
        block.jday = self.datetime.timetuple().tm_yday
        block.month = self.datetime.timetuple().tm_mon

        #block.latitude = self.h5['navigation']['latitudes'][SY, SX]  #original h5 code
        # block.longitude = self.h5['navigation']['longitudes'][SY, SX]  #original h5 code
        #block.sza = self.h5['navigation']['solar_zenith'][SY, SX]  #original h5 code
        #block.vza = self.h5['navigation']['sensor_zenith'][SY, SX]  #original h5 code
        #block.saa = self.h5['navigation']['solar_azimuth'][SY, SX]  #original h5 code
        #block.vaa = self.h5['navigation']['sensor_azimuth'][SY, SX]  #original h5 code

        block.latitude = self.nc.groups['navigation']['latitudes'][SY, SX]
        block.longitude = self.nc.groups['navigation']['longitudes'][SY, SX]
        block.sza = self.nc.groups['navigation']['solar_zenith'][SY, SX]
        block.vza = self.nc.groups['navigation']['sensor_zenith'][SY, SX]
        block.saa = self.nc.groups['navigation']['solar_azimuth'][SY, SX]
        block.vaa = self.nc.groups['navigation']['sensor_azimuth'][SY, SX]
        
        #assert len(self.Lt.attrs['wavelengths']) == len(bands_hico) #original h5 code
        assert len(self.Lt.getncattr('wavelengths')) == len(bands_hico)
        ibands = np.array([bands_hico.index(b) for b in bands])

        # read TOA
        block.Ltoa = np.zeros(size3) + np.NaN
        #slope = self.Lt.attrs['slope']  #original h5 code
        #intercept = self.Lt.attrs['intercept']  #original h5 code
        slope = self.Lt.getncattr('scale_factor')
        intercept= self.Lt.getncattr('add_offset')
        assert intercept == 0.
        block.Ltoa = slope * self.Lt[SY, SX, ibands]/10.  # convert

        # read bitmask
        block.bitmask = np.zeros(size, dtype='uint16')
        # flags = self.h5['quality']['flags'][SY, SX]
        # raiseflag(block.bitmask, L2FLAGS['LAND'], flags & 1 != 0)

        # read solar irradiance
        block.F0 = np.zeros(size3) + np.NaN
        block.F0[:,:,:] = F0_hico[None,None,ibands]

        # wavelength
        block.wavelen = np.zeros(size3, dtype='float32') + np.NaN
        block.wavelen[:,:,:] = wav_hico[None,None,ibands]
        block.cwavelen = wav_hico[ibands]

        # ancillary data
        block.ozone = np.zeros(size, dtype='float32')
        block.ozone[:] = self.ozone[block.latitude, block.longitude]
        block.wind_speed = np.zeros(size, dtype='float32')
        block.wind_speed[:] = self.wind_speed[block.latitude, block.longitude]
        block.surf_press = np.zeros(size, dtype='float32')
        block.surf_press[:] = self.surf_press[block.latitude, block.longitude]

        block.altitude = np.zeros(size, dtype='float32')

        if self.landmask is not None:
            raiseflag(block.bitmask, L2FLAGS['LAND'],
                      self.landmask_data[
                          yoffset+self.sline:yoffset+self.sline+ysize,
                          xoffset+self.scol:xoffset+self.scol+xsize,
                                         ])

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
        return OrderedDict()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass



