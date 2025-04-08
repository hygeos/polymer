#!/usr/bin/env python
# -*- coding: utf-8 -*-


from collections import OrderedDict
from pathlib import Path
from glymur import Jp2k
from glob import glob
from lxml import objectify
from os.path import join, dirname, exists
from datetime import datetime
import numpy as np
from polymer.block import Block
from polymer.utils import rectBivariateSpline
import pyproj
import pandas as pd
from polymer.ancillary import Ancillary_NASA
from polymer.common import L2FLAGS
from polymer.level1 import Level1_base
from polymer.utils import raiseflag
import xarray as xr

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

# https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/radiometric
msi_snr = {
    443: 129, 490: 154,
    560: 168, 665: 142,
    705: 117, 740: 89,
    783: 105, 842: 174,
    865: 72,  945: 114,
    1375: 50, 1610: 100,
    2190: 100,
}

msi_Lref = { # reference radiance
    443: 129 , 490: 128,
    560: 128 , 665: 108,
    705: 74.5, 740: 68,
    783: 67  , 842: 103,
    865: 52.5, 945: 9,
    1375: 6  , 1610: 4,
    2190: 1.5,
}


class Level1_MSI(Level1_base):

    def __init__(self, dirname, blocksize=198, resolution='60',
                 sline=0, eline=-1, scol=0, ecol=-1,
                 ancillary=None,
                 landmask=None,
                 altitude=0.,
                 srf_file=None,
                 use_srf=True,
                 add_noise=None,
                 ):
        '''
        Sentinel-2 MSI Level1 reader

        dirname: granule directory. Examples:
        * S2A_OPER_PRD_MSIL1C_20160318T145513.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20160318T232756_A003854_T19LDC_N02.01/
            (as downloaded on https://scihub.copernicus.eu/)
        * L1C_T51RTQ_A010954_20170728T024856/
            (as downloaded with Sentinelhub: https://github.com/sentinel-hub/sentinelhub-py)

        resolution: 60, 20 or 10 (in m)

        sline, eline, scol, ecol refers to the coordinate of the area to process:
            * in the 1830x1830 grid at 60m resolution
            * in the 5490x5490 grid at 20m resolution
              => eline-sline and ecol-scol must be a multiple of 3
            * in the 10980x10980 grid at 10m resolution
              => eline-sline and ecol-scol must be a multiple of 6

        ancillary: an ancillary data instance (Ancillary_NASA, Ancillary_ERA)
            or 'ECMWFT' for embedded ancillary data.

        landmask:
            * None: no land mask [default]
            * A GSW instance (see gsw.py)
              Example: landmask=GSW(directory='/path/to/gsw_data/')

        altitude: surface altitude in m
            * a float
            * a DEM instance such as:
                SRTM3(cache_dir=...)  # srtm.py
                GLOBE(directory=...)  # globe.py
                SRTM3(..., missing=GLOBE(...))

        srf_file: spectral response function. By default, it will use:
            auxdata/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2A.csv for S2A
            auxdata/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2B.csv for S2B
            auxdata/msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2C.csv for S2C

        use_srf: whether to calculate the bands central wavelengths from the SRF or to use fixed ones

        add_noise: function (band, reflectance, sza) -> stdev_reflectance
        '''

        # self.sensor = 'MSI'
        fname = Path(dirname).parent.parent.name
        if fname.startswith('S2A_MSIL1C') and fname.endswith('.SAFE'):
            self.sensor = 'MSIA'
        elif fname.startswith('S2B_MSIL1C') and fname.endswith('.SAFE'):
            self.sensor = 'MSIB'
        else:
            self.sensor = 'MSI'

        dirname = Path(dirname).resolve()
        if list(dirname.glob('GRANULE')):
            granules = list((dirname/'GRANULE').glob('*'))
            assert len(granules) == 1
            self.granule_dir = granules[0]
        else:
            self.granule_dir = dirname
        rootdir = self.granule_dir.parent.parent

        self.filename = str(rootdir)
        self.blocksize = blocksize
        self.resolution = str(resolution)
        self.landmask = landmask
        self.srf_file = srf_file
        self.altitude = altitude
        self.use_srf = use_srf
        self.sigma_typ = {k: msi_Lref[k]/msi_snr[k]
                          for k in msi_snr}
        self.Ltyp = msi_Lref
        self.add_noise = add_noise

        if ancillary is None:
            self.ancillary = Ancillary_NASA()
        else:
            self.ancillary = ancillary

        self.band_names = {
                443 : 'B01', 490 : 'B02',
                560 : 'B03', 665 : 'B04',
                705 : 'B05', 740 : 'B06',
                783 : 'B07', 842 : 'B08',
                865 : 'B8A', 945 : 'B09',
                1375: 'B10', 1610: 'B11',
                2190: 'B12',
                }

        # load xml file (granule)
        xmlfiles = glob(join(self.granule_dir, '*.xml'))
        assert len(xmlfiles) == 1
        xmlfile = xmlfiles[0]
        self.xmlgranule = objectify.parse(xmlfile).getroot()

        # load xml file (root)
        xmlfile = rootdir/'MTD_MSIL1C.xml'
        xmlroot = objectify.parse(str(xmlfile)).getroot()

        self.product_image_characteristics = xmlroot.General_Info.find('Product_Image_Characteristics')
        self.quantif = float(self.product_image_characteristics.QUANTIFICATION_VALUE)
        self.processing_baseline = xmlroot.General_Info.find('Product_Info').PROCESSING_BASELINE.text
        if float(self.processing_baseline) >= 4:
            self.radio_offset_list = [
                int(x)
                for x in self.product_image_characteristics.Radiometric_Offset_List.RADIO_ADD_OFFSET]
        else:
            self.radio_offset_list = [0]*len(self.band_names)

        self.date = datetime.strptime(str(self.xmlgranule.General_Info.find('SENSING_TIME')), '%Y-%m-%dT%H:%M:%S.%fZ')
        self.geocoding = self.xmlgranule.Geometric_Info.find('Tile_Geocoding')
        self.tileangles = self.xmlgranule.Geometric_Info.find('Tile_Angles')

        # get platform
        self.tile_id = str(self.xmlgranule.General_Info.find('TILE_ID')[0])
        self.platform = self.tile_id[:3]
        assert self.platform in ['S2A', 'S2B', 'S2C']

        # read image size for current resolution
        for e in self.geocoding.findall('Size'):
            if e.attrib['resolution'] == str(resolution):
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
        if self.ancillary == 'ECMWFT':
            self.init_ancillary_embedded()
        else:
            self.init_ancillary()
        self.init_landmask()
        self.init_bands()


    def init_bands(self):
        """ calculate equivalent wavelength from SRF """

        if self.srf_file is None:
            dir_aux_msi = join(dirname(dirname(__file__)), 'auxdata', 'msi')
            srf_file = join(dir_aux_msi, 'S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_{}.csv'.format(self.platform))
        else:
            srf_file = self.srf_file
        assert exists(srf_file)

        srf_data = pd.read_csv(srf_file)

        wav = srf_data.SR_WL

        self.wav = OrderedDict()
        for b, bn in self.band_names.items():
            col = self.platform + '_SR_AV_' + bn.replace('B0', 'B')
            srf = srf_data[col]
            # tauray = rod(wav/1000., 400., 45., 0., 1013.25)
            wav_eq = np.trapz(wav*srf)/np.trapz(srf)
            self.wav[b] = wav_eq
            # print(list(self.band_names.keys())[i], wav_eq)

    def init_ancillary(self):
        self.ozone = self.ancillary.get('ozone', self.date)
        self.wind_speed = self.ancillary.get('wind_speed', self.date)
        self.surf_press = self.ancillary.get('surf_press', self.date)

    def init_ancillary_embedded(self):
        file_auxdata = (self.granule_dir/'AUX_DATA'/'AUX_ECMWFT')
        ds = xr.open_dataset(file_auxdata, engine="cfgrib")
        self.wind_speed = np.sqrt(ds.u10**2 + ds.v10**2)
        assert ds.tco3.units == 'kg m**-2'
        self.ozone = ds.tco3/2.1415e-5  # convert kg/m2 to DU
        assert ds.msl.units == 'Pa'
        self.surf_press = ds.msl/100
                

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

        X, Y = np.meshgrid(ULX + XDIM//2 + XDIM*np.arange(self.totalheight), 
                           ULY + YDIM//2 + YDIM*np.arange(self.totalwidth))

        self.lon, self.lat = (proj(X, Y, inverse=True))


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

    def init_landmask(self):
        if not hasattr(self.landmask, 'get'):
            return

        self.landmask_data = self.landmask.get(self.lat, self.lon)


    def get_filename(self, band):
        '''
        returns jp2k filename containing band
        '''
        filenames = glob(join(self.granule_dir, 'IMG_DATA/*_{}.jp2'.format(self.band_names[band])))
        assert len(filenames) == 1

        return filenames[0]


    def read_TOA(self, band, size, offset):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset

        jp = Jp2k(self.get_filename(band))

        xrat = jp.shape[0]/float(self.totalwidth)
        yrat = jp.shape[1]/float(self.totalheight)

        # read input data
        datao = jp[
           int(yrat*(yoffset+self.sline)): int(yrat*(yoffset+self.sline+ysize)),
           int(xrat*(xoffset+self.scol)) : int(xrat*(xoffset+self.scol+xsize))
           ]

        if xrat >= 1.:
            # downsample
            data = np.zeros(size, dtype='float32')
            N = 0
            for i in range(int(xrat)):
                for j in range(int(yrat)):
                    N += 1
                    data += datao[j::int(yrat), i::int(xrat)]
            data /= N
        else:
            # over-sample
            data = datao.repeat(int(1/yrat), axis=0).repeat(int(1/xrat), axis=1)

        assert data.shape == size, '{} != {}'.format(data.shape, size)

        return data

    def read_block(self, size, offset, bands):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset
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

        block.bitmask = np.zeros(size, dtype='uint16')
        raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], np.isnan(block.muv))

        # read RTOA
        block.Rtoa = np.zeros((ysize,xsize,nbands)) + np.nan
        for iband, band in enumerate(bands):
            raw_data = self.read_TOA(band, size, offset)
            if iband == 0:
                raiseflag(block.bitmask, L2FLAGS['L1_INVALID'], raw_data == 0)

            Rtoa = (raw_data + self.radio_offset_list[iband])/self.quantif

            if self.add_noise is not None:
                stdev = self.add_noise(band=band,
                                       Rtoa=Rtoa,
                                       sza=block.sza)
                noise = stdev*np.random.normal(0, 1, stdev.size).reshape(stdev.shape)
            else:
                noise = 0

            block.Rtoa[:,:,iband] = Rtoa + noise

        if self.landmask is not None:
            raiseflag(block.bitmask, L2FLAGS['LAND'],
                      self.landmask_data[
                          yoffset+self.sline:yoffset+self.sline+ysize,
                          xoffset+self.scol:xoffset+self.scol+xsize,
                                         ])

        block.wavelen = np.zeros((ysize, xsize, nbands), dtype='float32') + np.nan
        block.cwavelen = np.zeros(nbands, dtype='float32') + np.nan
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = self.wav[band]
            block.cwavelen[iband] = self.wav[band]

        # read surface altitude
        try:
            block.altitude = self.altitude.get(lat=block.latitude,
                                               lon=block.longitude)
        except AttributeError:
            # altitude expected to be a float
            block.altitude = np.zeros((ysize, xsize), dtype='float32') + self.altitude

        if self.ancillary == 'ECMWFT':
            # ancillary data embedded in Level1
            block.ozone = self.ozone.interp(
                latitude=xr.DataArray(block.latitude),
                longitude=xr.DataArray(block.longitude),
                # method='nearest'
                ).values
            block.wind_speed = self.wind_speed.interp(
                latitude=xr.DataArray(block.latitude),
                longitude=xr.DataArray(block.longitude),
                # method='nearest'
                ).values
            P0 = self.surf_press.interp(
                latitude=xr.DataArray(block.latitude),
                longitude=xr.DataArray(block.longitude),
                # method='nearest'
                ).values
        else:
            block.ozone = self.ozone[block.latitude, block.longitude]
            block.wind_speed = self.wind_speed[block.latitude, block.longitude]
            P0 = self.surf_press[block.latitude, block.longitude]

        block.surf_press = P0 * np.exp(-block.altitude/8000.)

        if not self.use_srf:
            block.tau_ray = np.zeros((ysize, xsize, nbands), dtype='float32') + np.nan
            for iband, band in enumerate(bands):
                block.tau_ray[:,:,iband] = {  # first calculations
                                              # using convolution of ROD using old version of SRF
                            443: 0.234280641095, 490: 0.149710335414,
                            560: 0.0905014442489, 665: 0.0450755094234,
                            705: 0.0356256787869, 740: 0.0290586201022,
                            783: 0.0232262166776, 842: 0.0181555100059,
                            865: 0.0155121863123, 940: 0.0108505370873,
                            1375: 0.00242549670396, 1610: 0.00128165077197,
                            2190: 0.000383201294006,
                        }[band] * block.surf_press/1013.

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
        attr['sensing_time'] = self.date.strftime(datefmt)
        attr['L1_TILE_ID'] = str(self.xmlgranule.General_Info.find('TILE_ID'))
        attr['L1_DATASTRIP_ID'] = str(self.xmlgranule.General_Info.find('DATASTRIP_ID'))
        attr['central_wavelength'] = str(dict(self.wav))
        attr['processing_baseline'] = str(self.processing_baseline)
        return attr


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


