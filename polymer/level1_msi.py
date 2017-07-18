#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division, absolute_import
from glymur import Jp2k
from glob import glob
from lxml import objectify
from os.path import join
from datetime import datetime
import numpy as np
from polymer.block import Block
from polymer.utils import rectBivariateSpline
import pyproj
from polymer.ancillary import Ancillary_NASA
from polymer.common import L2FLAGS
from polymer.level1 import Level1_base

'''
List of MSI bands:
-----------------

Band Use         Wavelength Resolution
B1   Aerosols    443nm      60m
B2   Blue        490nm      10m
B3   Green       560nm      10m
B4   Red         665nm      10m
B5   Red Edge 1  705nm      20m
B6   Red Edge 2  740nm      20m
B7   Red Edge 3  783nm      20m
B8   NIR         842nm      10m
B8a  Red Edge 4  865nm      20m
B9   Water vapor 940nm      60m
B10  Cirrus      1375nm     60m
B11  SWIR 1      1610nm     20m
B12  SWIR 2      2190nm     20m
'''


class Level1_MSI(Level1_base):

    def __init__(self, dirname, blocksize=200, resolution='60',
                 sline=0, eline=-1, scol=0, ecol=-1,
                 ancillary=None):
        '''
        dirname: granule dirname

        resolution: 60, 20 or 10m
        '''
        self.sensor = 'MSI'
        self.dirname = dirname
        self.filename = dirname
        self.blocksize = blocksize
        self.resolution = resolution
        assert isinstance(resolution, str)

        if ancillary is None:
            self.ancillary = Ancillary_NASA()
        else:
            self.ancillary = ancillary

        self.band_names = {
                443 : 'B01', 490 : 'B02',
                560 : 'B03', 665 : 'B04',
                705 : 'B05', 740 : 'B06',
                783 : 'B07', 842 : 'B08',
                865 : 'B8A', 940 : 'B09',
                1375: 'B10', 1610: 'B11',
                2190: 'B12',
                }

        # load xml file
        xmlfiles = glob(join(self.dirname, '*.xml'))
        assert len(xmlfiles) == 1
        xmlfile = xmlfiles[0]
        self.xmlroot = objectify.parse(xmlfile).getroot()
        self.date = datetime.strptime(str(self.xmlroot.General_Info.find('SENSING_TIME')), '%Y-%m-%dT%H:%M:%S.%fZ')
        self.geocoding = self.xmlroot.Geometric_Info.find('Tile_Geocoding')
        self.tileangles = self.xmlroot.Geometric_Info.find('Tile_Angles')

        # read image size for current resolution
        for e in self.geocoding.findall('Size'):
            if e.attrib['resolution'] == resolution:
                totalheight = int(e.find('NROWS').text)
                totalwidth = int(e.find('NCOLS').text)
                break

        self.init_shape(
                totalheight=totalheight,
                totalwidth=totalwidth,
                sline=sline,
                eline=eline,
                scol=scol,
                ecol=ecol)

        self.init_latlon()
        self.init_geometry()
        self.init_ancillary()

    def init_ancillary(self):
        self.ozone = self.ancillary.get('ozone', self.date)
        self.wind_speed = self.ancillary.get('wind_speed', self.date)
        self.surf_press = self.ancillary.get('surf_press', self.date)


    def init_latlon(self):

        code = self.geocoding.find('HORIZONTAL_CS_CODE').text

        print('Initialize MSI projection {}'.format(code))

        proj = pyproj.Proj('+init={}'.format(code))

        # lookup position in the UTM grid
        for e in self.geocoding.findall('Geoposition'):
            if e.attrib['resolution'] == self.resolution:
                ULX = int(e.find('ULX').text)
                ULY = int(e.find('ULY').text)
                XDIM = int(e.find('XDIM').text)
                YDIM = int(e.find('YDIM').text)

        X, Y = np.meshgrid(ULX + XDIM*np.arange(self.totalheight), 
                           ULY + YDIM*np.arange(self.totalwidth))

        self.lon, self.lat = (proj(X, Y, inverse=True))

        # TODO: what about -180 -> 180 ???

    def init_geometry(self):

        # read solar angles at tiepoints
        sza = read_xml_block(self.tileangles.find('Sun_Angles_Grid').find('Zenith').find('Values_List'))
        saa = read_xml_block(self.tileangles.find('Sun_Angles_Grid').find('Azimuth').find('Values_List'))

        shp = (self.totalheight, self.totalwidth)

        self.sza = rectBivariateSpline(sza, shp)
        self.saa = rectBivariateSpline(saa, shp)

        # read view angles (for each band)
        vza = {}
        vaa = {}
        for e in self.tileangles.findall('Viewing_Incidence_Angles_Grids'):

           # read zenith angles
           data = read_xml_block(e.find('Zenith').find('Values_List'))
           bandid = int(e.attrib['bandId'])
           if not bandid in vza:
               vza[bandid] = data
           else:
               ok = ~np.isnan(data)
               vza[bandid][ok] = data[ok]

           # read azimuth angles
           data = read_xml_block(e.find('Azimuth').find('Values_List'))
           bandid = int(e.attrib['bandId'])
           if not bandid in vaa:
               vaa[bandid] = data
           else:
               ok = ~np.isnan(data)
               vaa[bandid][ok] = data[ok]

        self.vza = np.zeros(shp, dtype='float32')
        self.vaa = np.zeros(shp, dtype='float32')

        # use the first band as vza and vaa
        k = sorted(vza.keys())[0]
        assert k in vaa
        self.vza[:,:] = rectBivariateSpline(vza[k], shp)
        self.vaa[:,:] = rectBivariateSpline(vaa[k], shp)

    def get_filename(self, band):
        '''
        returns jp2k filename containing band
        '''
        filenames = glob(join(self.dirname, 'IMG_DATA/*_{}.jp2'.format(self.band_names[band])))
        assert len(filenames) == 1

        return filenames[0]


    def read_TOA(self, band, size, offset):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset

        jp = Jp2k(self.get_filename(band))

        xrat = jp.shape[0]//self.totalwidth
        yrat = jp.shape[1]//self.totalheight

        # read the whole array that will be downsampled
        datao = jp[
           yrat*(yoffset+self.sline) : yrat*(yoffset+self.sline+ysize),
           xrat*(xoffset+self.scol) : xrat*(xoffset+self.scol+xsize)
           ]

        data = np.zeros(size, dtype='float32')
        N = 0
        for i in range(xrat):
            for j in range(yrat):
                N += 1
                data += datao[j::yrat, i::xrat]

        data /= N

        return data

    def read_block(self, size, offset, bands):

        (ysize, xsize) = size
        nbands = len(bands)

        block = Block(offset=offset, size=size, bands=bands)

        SY = slice(offset[0]+self.sline, offset[0]+self.sline+ysize)
        SX = slice(offset[1]+self.scol, offset[1]+self.scol+xsize)

        # read lat/lon
        block.latitude = self.lat[SY, SX]
        block.longitude = self.lon[SY, SX]

        # read geometry
        block.sza = self.sza[SY, SX]
        block.saa = self.saa[SY, SX]

        block.vza = self.vza[SY, SX]
        block.vaa = self.vaa[SY, SX]

        block.jday = self.date.timetuple().tm_yday
        block.month = self.date.timetuple().tm_mon

        # read RTOA
        block.Rtoa = np.zeros((ysize,xsize,nbands)) + np.NaN
        for iband, band in enumerate(bands):
            QUANTIF = 10000
            block.Rtoa[:,:,iband] = self.read_TOA(band, size, offset)/QUANTIF

        block.bitmask = np.zeros(size, dtype='uint16')
        # very crude land mask
        # block.bitmask += L2FLAGS['LAND']*landmask(
                # block.latitude, block.longitude,
                # resolution='f').astype('uint16')
        block.bitmask += L2FLAGS['L1_INVALID']*(np.isnan(block.muv).astype('uint16'))

        block.ozone = self.ozone[block.latitude, block.longitude]
        block.wind_speed = self.wind_speed[block.latitude, block.longitude]
        block.surf_press = self.surf_press[block.latitude, block.longitude]

        block.wavelen = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = float(band)

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
        return {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def read_xml_block(item):
    '''
    read a block of xml data and returns it as a numpy float32 array
    '''
    d = []
    for i in item.iterchildren():
        d.append(i.text.split())
    return np.array(d, dtype='float32')


