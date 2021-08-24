#!/usr/bin/env python
# -*- coding: utf-8 -*-

from polymer.level1 import Level1_base
from polymer.block import Block
from polymer.ancillary import Ancillary_NASA
from polymer.level1_landsat8_meta import read_meta
from polymer.utils import raiseflag
from polymer.common import L2FLAGS
from datetime import datetime, time
from collections import OrderedDict
from osgeo import gdal
import osgeo
from osgeo import osr
from glob import glob
import numpy as np
import os
from os.path import dirname, join
import xlrd

gdal_major_version = int(osgeo.__version__.split('.')[0])

band_index = {
        440: 1,
        480: 2,
        560: 3,
        655: 4,
        865: 5,
        1610: 6,
        2200: 7,
        }


class Level1_OLI(Level1_base):
    '''
    Landsat-8 OLI data reader

    altitude: surface altitude in m
        * a float
        * a DEM instance such as:
            SRTM3(cache_dir=...)  # srtm.py
            GLOBE(directory=...)  # globe.py
            SRTM3(..., missing=GLOBE(...))
    landmask:
        * None: no land mask [default]
        * A GSW instance (see gsw.py)
          Example: landmask=GSW(directory='/path/to/gsw_data/')

    Note: requires angle files generated with:
        l8_angles LC08_..._ANG.txt BOTH 1 -b 1

        l8_angles is available at:
        https://www.usgs.gov/land-resources/nli/landsat/solar-illumination-and-sensor-viewing-angle-coefficient-files
    '''
    def __init__(self, dirname,
                 sline=0, eline=-1,
                 scol=0, ecol=-1, ancillary=None,
                 altitude=0.,
                 landmask=None,
                 blocksize=(500, 400)):
        # https://stackoverflow.com/questions/2922532
        self.sensor = 'OLI'
        self.blocksize = blocksize
        self.altitude = altitude
        self.dirname = dirname
        self.filename = dirname
        self.landmask = landmask
        if ancillary is None:
            self.ancillary = Ancillary_NASA()
        else:
            self.ancillary = ancillary

        files_B1 = glob(os.path.join(dirname, 'LC*_B1.TIF'))
        if len(files_B1) != 1:
            raise Exception('Invalid directory content ({})'.format(files_B1))
        file_B1 = files_B1[0]

        print('Reading coordinates from {}'.format(file_B1))

        ds = gdal.Open(file_B1)
        old_cs= osr.SpatialReference()
        if gdal_major_version >= 3:
            # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
            old_cs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        old_cs.ImportFromWkt(ds.GetProjectionRef())

        # create the new coordinate system
        wgs84_wkt = """
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]"""
        new_cs = osr.SpatialReference()
        if gdal_major_version >= 3:
            # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
            new_cs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        new_cs.ImportFromWkt(wgs84_wkt)

        # create a transform object to convert between coordinate systems
        self.transform = osr.CoordinateTransformation(old_cs, new_cs)

        # get the point to transform, pixel (0,0) in this case
        width = ds.RasterXSize
        height = ds.RasterYSize
        self.init_shape(
                totalheight=height,
                totalwidth=width,
                sline=sline,
                eline=eline,
                scol=scol,
                ecol=ecol)
        gt = ds.GetGeoTransform()

        X0,X1 = (0, width-1)
        Y0,Y1 = (0, height-1)

        print('Image size is {}x{}'.format(width, height))

        # generally:
        # Xgeo0 = gt[0] + X0*gt[1] + Y0*gt[2]
        # Ygeo0 = gt[3] + X0*gt[4] + Y0*gt[5]
        # Xgeo1 = gt[0] + X1*gt[1] + Y0*gt[2]
        # Ygeo1 = gt[3] + X1*gt[4] + Y0*gt[5]
        # Ygeo2 = gt[3] + X0*gt[4] + Y1*gt[5]
        # Xgeo2 = gt[0] + X0*gt[1] + Y1*gt[2]
        # Xgeo3 = gt[0] + X1*gt[1] + Y1*gt[2]
        # Ygeo3 = gt[3] + X1*gt[4] + Y1*gt[5]

        # this simplifies because gt[2] == gt[4] == 0
        assert gt[2] == 0
        assert gt[4] == 0
        Xmin = gt[0] + X0*gt[1]
        Xmax = gt[0] + X1*gt[1]
        Ymin = gt[3] + Y0*gt[5]
        Ymax = gt[3] + Y1*gt[5]

        XY = np.array(np.meshgrid(
            np.linspace(Xmin, Xmax, self.totalwidth)[self.scol:self.scol+self.width],
            np.linspace(Ymin, Ymax, self.totalheight)[self.sline:self.sline+self.height],
            ))
        XY = np.moveaxis(XY, 0, -1)

        # get the coordinates in lat long
        latlon = np.array(self.transform.TransformPoints(XY.reshape((-1, 2))))
        self.lat = latlon[:,1].reshape((self.height, self.width))
        self.lon = latlon[:,0].reshape((self.height, self.width))

        # Initializations
        self.init_meta()
        self.init_ancillary()
        self.init_spectral()
        self.init_landmask()
        self.init_geometry()


    def date(self):
        d = self.attr_date['DATE_ACQUIRED']
        t = datetime.strptime(self.attr_date['SCENE_CENTER_TIME'][:8], '%H:%M:%S')
        return datetime.combine(d, time(t.hour, t.minute, t.second))

    def init_ancillary(self):
        self.ozone = self.ancillary.get('ozone', self.date())
        self.wind_speed = self.ancillary.get('wind_speed', self.date())
        self.surf_press = self.ancillary.get('surf_press', self.date())

        self.ancillary_files = OrderedDict()
        self.ancillary_files.update(self.ozone.filename)
        self.ancillary_files.update(self.wind_speed.filename)
        self.ancillary_files.update(self.surf_press.filename)

    def init_meta(self):
        # read metadata
        files_mtl = glob(os.path.join(self.dirname, 'LC*_MTL.txt'))
        assert len(files_mtl) == 1
        file_mtl = files_mtl[0]
        meta = read_meta(file_mtl)
        if 'L1_METADATA_FILE' in meta:
            # Collection 1
            self.level1_meta = meta['L1_METADATA_FILE']
            self.attr_product = self.level1_meta['PRODUCT_METADATA']
            self.file_ang = self.attr_product['ANGLE_COEFFICIENT_FILE_NAME']
            self.attr_rescaling = self.level1_meta['RADIOMETRIC_RESCALING']
            self.attr_date = self.level1_meta['PRODUCT_METADATA']
        else:
            # Collection 2
            self.level1_meta = meta['LANDSAT_METADATA_FILE']
            self.attr_product = self.level1_meta['PRODUCT_CONTENTS']
            self.file_ang = self.attr_product['FILE_NAME_ANGLE_COEFFICIENT']
            self.attr_rescaling = self.level1_meta['LEVEL1_RADIOMETRIC_RESCALING']
            self.attr_date = self.level1_meta['IMAGE_ATTRIBUTES']

        # read angles file
        file_ang = os.path.join(
                self.dirname,
                self.file_ang
                )
        self.data_ang = read_meta(file_ang)


    def init_spectral(self):
        dir_aux_oli = join(dirname(dirname(__file__)), 'auxdata', 'oli')
        srf_file = join(dir_aux_oli, 'Ball_BA_RSR.v1.2.xlsx')

        wb = xlrd.open_workbook(srf_file)


        self.wav = OrderedDict()
        for b, bname in [(440, 'CoastalAerosol'),
                         (480, 'Blue'),
                         (560, 'Green'),
                         (655, 'Red'),
                         (865, 'NIR'),
                         (1610, 'SWIR1'),
                         (2200, 'SWIR2')]:
            sh = wb.sheet_by_name(bname)

            wav, srf = [], []

            i=0
            while True:
                i += 1
                try:
                    wav.append(sh.cell(i, 0).value)
                    srf.append(sh.cell(i, 1).value)
                except IndexError:
                    break

            wav, srf = np.array(wav), np.array(srf)
            wav_eq = np.trapz(wav*srf)/np.trapz(srf)
            self.wav[b] = wav_eq


    def init_geometry(self):
        # read sensor angles
        filenames_sensor = glob(join(self.dirname, 'LC*_sensor_B01.img'))
        assert len(filenames_sensor) == 1, 'Error, sensor angles file missing ({})'.format(str(filenames_sensor))
        filename_sensor = filenames_sensor[0]
        self.data_sensor = np.fromfile(filename_sensor, dtype='int16').astype('float32').reshape((2, self.totalheight, self.totalwidth))/100.

        # read solar angles
        filenames_solar = glob(join(self.dirname, 'LC*_solar_B01.img'))
        assert len(filenames_solar) == 1, 'Error, solar angles file missing ({})'.format(str(filenames_solar))
        filename_solar = filenames_solar[0]
        self.data_solar = np.fromfile(filename_solar, dtype='int16').astype('float32').reshape((2, self.totalheight, self.totalwidth))/100.


    def init_landmask(self):
        if not hasattr(self.landmask, 'get'):
            return

        self.landmask_data = self.landmask.get(self.lat, self.lon)


    def read_block(self, size, offset, bands):
        (ysize, xsize) = size
        (yoffset, xoffset) = offset
        nbands = len(bands)
        block = Block(offset=offset, size=size, bands=bands)

        SY = slice(offset[0]+self.sline, offset[0]+self.sline+ysize)
        SX = slice(offset[1]+self.scol, offset[1]+self.scol+xsize)

        # get the coordinates in lat long
        block.latitude = self.lat[offset[0]:offset[0]+ysize, offset[1]:offset[1]+xsize]
        block.longitude = self.lon[offset[0]:offset[0]+ysize, offset[1]:offset[1]+xsize]

        # geometry
        block.sza = self.data_solar[1, SY, SX]
        block.vza = self.data_sensor[1,SY, SX]
        block.saa = self.data_solar[0, SY, SX]
        block.vaa = self.data_sensor[0,SY, SX]

        # TOA reflectance
        block.Rtoa = np.zeros((ysize,xsize,nbands)) + np.NaN
        for iband, band in enumerate(bands):
            M = self.attr_rescaling['REFLECTANCE_MULT_BAND_{}'.format(band_index[band])]
            A = self.attr_rescaling['REFLECTANCE_ADD_BAND_{}'.format(band_index[band])]

            filename = os.path.join(
                    self.dirname,
                    self.attr_product['FILE_NAME_BAND_{}'.format(band_index[band])])
            dset = gdal.Open(filename)
            band = dset.GetRasterBand(1)
            data = band.ReadAsArray(xoff=self.scol+xoffset, yoff=self.sline+yoffset,
                                    win_xsize=xsize, win_ysize=ysize)
            block.Rtoa[:,:,iband] = (M*data + A)/np.cos(np.radians(block.sza))
            block.Rtoa[:,:,iband][data == 0] = np.NaN


        # spectral info
        block.wavelen = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        block.cwavelen = np.zeros(nbands, dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = self.wav[band]
            block.cwavelen[iband] = self.wav[band]

        # date
        block.jday = self.date().timetuple().tm_yday
        block.month = self.date().timetuple().tm_mon

        # ancillary data
        block.ozone = self.ozone[block.latitude, block.longitude]
        block.wind_speed = self.wind_speed[block.latitude, block.longitude]
        P0 = self.surf_press[block.latitude, block.longitude]

        # read surface altitude
        try:
            block.altitude = self.altitude.get(lat=block.latitude,
                                               lon=block.longitude)
        except AttributeError:
            # altitude expected to be a float
            block.altitude = np.zeros((ysize, xsize), dtype='float32') + self.altitude

        # calculate surface altitude
        block.surf_press = P0 * np.exp(-block.altitude/8000.)

        # quality flags
        block.bitmask = np.zeros(size, dtype='uint16')

        # landmask
        if self.landmask is not None:
            raiseflag(block.bitmask, L2FLAGS['LAND'],
                      self.landmask_data[
                          yoffset:yoffset+ysize,
                          xoffset:xoffset+xsize,
                                         ])
        # invalid level1
        raiseflag(block.bitmask, L2FLAGS['L1_INVALID'],
                  np.isnan(block.Rtoa[:,:,0]))

        return block

    def attributes(self, datefmt):
        attr = OrderedDict()
        attr['l1_dirname'] = self.dirname
        return attr

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
