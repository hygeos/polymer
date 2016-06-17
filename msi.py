#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, division
from glymur import Jp2k
from glob import glob
from lxml import objectify
from os.path import join
import numpy as np
from params import Params
from block import Block
from utils import rectBivariateSpline
import pyproj

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


class Params_MSI(Params):

    def __init__(self, **kwargs):

        super(self.__class__, self).__init__()

        self.lut_file = '/home/francois/MERIS/POLYMER/LUTS/MSI/LUT.hdf'

        # FIXME
        self.bands_corr = [443,490,560,665,705,740,783,    865,         1610,    ]
        self.bands_oc   = [443,490,560,665,705,740,783,    865,         1610,    ]
        self.bands_rw   = [443,490,560,665,705,740,783,    865,         1610,    ]

        self.bands_lut =  [443,490,560,665,705,740,783,842,865,945,1375,1610,2190]
        self.central_wavelength = {
                443 : 443., 490 : 490.,
                560 : 560., 665 : 665.,
                705 : 705., 740 : 740.,
                783 : 783., 842 : 842.,
                865 : 865., 945 : 945.,
                1375: 1375., 1610: 1610.,
                2190: 2190.,
                }

        self.band_cloudmask = 865

        self.K_OZ = {   # FIXME
                443 : 0.,
                490 : 0.,
                560 : 0.,
                665 : 0.,
                705 : 0.,
                740 : 0.,
                783 : 0.,
                842 : 0.,
                865 : 0.,
                945 : 0.,
                1375: 0.,
                1610: 0.,
                2190: 0.,
                }

        self.K_NO2 = {   # FIXME
                443 : 0.,
                490 : 0.,
                560 : 0.,
                665 : 0.,
                705 : 0.,
                740 : 0.,
                783 : 0.,
                842 : 0.,
                865 : 0.,
                945 : 0.,
                1375: 0.,
                1610: 0.,
                2190: 0.,
                }

        self.update(**kwargs)


class Level1_MSI(object):

    def __init__(self, dirname, blocksize=100, resolution='60', sline=0, eline=-1):
        '''
        dirname: granule dirname

        resolution: 60, 20 or 10m
        '''
        self.dirname = dirname
        self.filename = dirname
        self.blocksize = blocksize
        self.resolution = resolution
        assert isinstance(resolution, str)
        self.sline = sline
        self.eline = eline

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
        xmlfile = glob(join(self.dirname, '*.xml'))[0]
        self.xmlroot = objectify.parse(xmlfile).getroot()
        self.geocoding = self.xmlroot.Geometric_Info.find('Tile_Geocoding')
        self.tileangles = self.xmlroot.Geometric_Info.find('Tile_Angles')

        # read image size for current resolution
        for e in self.geocoding.findall('Size'):
            if e.attrib['resolution'] == resolution:
                self.totalheight = int(e.find('NROWS').text)
                self.width = int(e.find('NCOLS').text)
                break

        if eline < 0:
            self.height = self.totalheight
            self.height -= sline
            self.height += eline + 1
        else:
            self.height = eline-sline

        self.shape = (self.height, self.width)

        self.init_latlon()
        self.init_geometry()


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

        X, Y = np.meshgrid(ULX + XDIM*np.arange(self.height), 
                           ULY + YDIM*np.arange(self.width))

        self.lon, self.lat = (proj(X, Y, inverse=True))

        # TODO: what about -180 -> 180 ???

    def init_geometry(self):

        # read solar angles at tiepoints
        sza = read_xml_block(self.tileangles.find('Sun_Angles_Grid').find('Zenith').find('Values_List'))
        saa = read_xml_block(self.tileangles.find('Sun_Angles_Grid').find('Azimuth').find('Values_List'))

        self.sza = rectBivariateSpline(sza, self.shape)
        self.saa = rectBivariateSpline(saa, self.shape)

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

        self.vza = np.zeros(self.shape+(len(self.band_names),),
                dtype='float32')
        self.vaa = np.zeros(self.shape+(len(self.band_names),),
                dtype='float32')
        for i in range(len(self.band_names)):
            self.vza[:,:,i] = rectBivariateSpline(vza[i], self.shape)
            self.vaa[:,:,i] = rectBivariateSpline(vaa[i], self.shape)


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

        xrat = jp.shape[0]//self.shape[0]
        yrat = jp.shape[1]//self.shape[1]

        # read the whole array that will be downsampled
        datao = jp[
           yrat*(yoffset+self.sline) : yrat*(yoffset+self.sline+ysize),
           xrat*xoffset : xrat*(xoffset+xsize)
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
        SX = slice(offset[1], offset[1]+xsize)

        # read lat/lon
        block.latitude = self.lat[SY, SX]
        block.longitude = self.lon[SY, SX]

        # read geometry
        block.sza = self.sza[SY, SX]
        block.saa = self.saa[SY, SX]

        # for the view angles we select that of band 0
        band_id = 0
        block.vza = self.vza[SY, SX, band_id]
        block.vaa = self.vaa[SY, SX, band_id]

        block.jday = 100  # FIXME

        # read RTOA
        block.Rtoa = np.zeros((ysize,xsize,nbands)) + np.NaN
        for iband, band in enumerate(bands):
            QUANTIF = 10000
            block.Rtoa[:,:,iband] = self.read_TOA(band, size, offset)/QUANTIF

        block.bitmask = np.zeros(size, dtype='uint16') # FIXME
        block.ozone = np.zeros(size, dtype='float32') + 300.  # FIXME
        block.wind_speed = np.zeros(size, dtype='float32') + 5.  # FIXME
        block.surf_press = np.zeros(size, dtype='float32') + 1013.   # FIXME

        block.wavelen = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = float(band)

        print('Read', block)

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


def read_xml_block(item):
    '''
    read a block of xml data and returns it as a numpy float32 array
    '''
    d = []
    for i in item.iterchildren():
        d.append(i.text.split())
    return np.array(d, dtype='float32')


