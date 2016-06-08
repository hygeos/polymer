#!/usr/bin/env python
# encoding: utf-8


import epr
from block import Block
import numpy as np
from datetime import datetime
import warnings
from params import Params
from common import L2FLAGS


class Params_MERIS(Params):
    '''
    MERIS-specific parameters
    '''
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__()

        self.bands_corr = [412,443,490,510,560,620,665,        754,    779,865]
        self.bands_oc =   [412,443,490,510,560,620,665,        754,    779,865]
        self.bands_rw =   [412,443,490,510,560,620,665,        754,    779,865]

        self.lut_file = '/home/francois/MERIS/POLYMER/LUTS/MERIS/LUTB.hdf'
        self.bands_lut = [412,443,490,510,560,620,665,681,709,754,760,779,865,885,900]

        self.band_cloudmask = 865

        self.K_OZ = {
                    412: 0.000301800 , 443: 0.00327200 ,
                    490: 0.0211900   , 510: 0.0419600  ,
                    560: 0.104100    , 620: 0.109100   ,
                    665: 0.0511500   , 681: 0.0359600  ,
                    709: 0.0196800   , 754: 0.00955800 ,
                    760: 0.00730400  , 779: 0.00769300 ,
                    865: 0.00219300  , 885: 0.00121100 ,
                    900: 0.00151600  ,
                }

        self.K_NO2 = {
                412: 6.074E-19 , 443: 4.907E-19,
                490: 2.916E-19 , 510: 2.218E-19,
                560: 7.338E-20 , 620: 2.823E-20,
                665: 6.626E-21 , 681: 6.285E-21,
                709: 4.950E-21 , 754: 1.384E-21,
                760: 4.717E-22 , 779: 3.692E-22,
                865: 2.885E-23 , 885: 4.551E-23,
                900: 5.522E-23 ,
                }

        self.central_wavelength = {
                412: 412.691 , 443: 442.559,
                490: 489.882 , 510: 509.819,
                560: 559.694 , 620: 619.601,
                665: 664.573 , 681: 680.821,
                709: 708.329 , 754: 753.371,
                760: 761.508 , 779: 778.409,
                865: 864.876 , 885: 884.944,
                900: 900.000 ,
                }

        self.NO2_CLIMATOLOGY = '/home/francois/MERIS/POLYMER/auxdata/common/no2_climatology.hdf'
        self.NO2_FRAC200M = '/home/francois/MERIS/POLYMER/auxdata/common/trop_f_no2_200m.hdf'

        # update 
        self.update(**kwargs)


class Level1_MERIS(object):

    def __init__(self, filename, sline=0, eline=-1, blocksize=50):

        self.filename = filename
        self.prod = epr.Product(filename)
        self.width = self.prod.get_scene_width()
        self.totalheight = self.prod.get_scene_height()
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
        self.band_names = {
                412: 'Radiance_1', 443: 'Radiance_2',
                490: 'Radiance_3', 510: 'Radiance_4',
                560: 'Radiance_5', 620: 'Radiance_6',
                665: 'Radiance_7', 681: 'Radiance_8',
                709: 'Radiance_9', 754: 'Radiance_10',
                760: 'Radiance_11', 779: 'Radiance_12',
                865: 'Radiance_13', 885: 'Radiance_14',
                900: 'Radiance_15',
            }

        # initialize solar irradiance
        self.F0 = np.genfromtxt('/home/francois/MERIS/POLYMER/auxdata/meris/smile/v2/sun_spectral_flux_rr.txt', names=True)
        self.F0_band_names = {
                    412: 'E0_band0', 443: 'E0_band1',
                    490: 'E0_band2', 510: 'E0_band3',
                    560: 'E0_band4', 620: 'E0_band5',
                    665: 'E0_band6', 681: 'E0_band7',
                    709: 'E0_band8', 754: 'E0_band9',
                    760: 'E0_band10', 779: 'E0_band11',
                    865: 'E0_band12', 885: 'E0_band13',
                    900: 'E0_band14',
                    }
        self.wav_band_names = {
                    412: 'lam_band0', 443: 'lam_band1',
                    490: 'lam_band2', 510: 'lam_band3',
                    560: 'lam_band4', 620: 'lam_band5',
                    665: 'lam_band6', 681: 'lam_band7',
                    709: 'lam_band8', 754: 'lam_band9',
                    760: 'lam_band10', 779: 'lam_band11',
                    865: 'lam_band12', 885: 'lam_band13',
                    900: 'lam_band14',
                }

        # initialize detector wavelength
        self.detector_wavelength = np.genfromtxt('/home/francois/MERIS/POLYMER/auxdata/meris/smile/v2/central_wavelen_rr.txt', names=True)   # FIXME

        # read the file date
        mph = self.prod.get_mph()
        dat = mph.get_field('SENSING_START').get_elem(0)
        dat = dat.replace('-JAN-', '-01-')  # NOTE:
        dat = dat.replace('-FEB-', '-02-')  # parsing with '%d-%b-%Y...' may be
        dat = dat.replace('-MAR-', '-03-')  # locale-dependent
        dat = dat.replace('-APR-', '-04-')
        dat = dat.replace('-MAY-', '-05-')
        dat = dat.replace('-JUN-', '-06-')
        dat = dat.replace('-JUL-', '-07-')
        dat = dat.replace('-AUG-', '-08-')
        dat = dat.replace('-SEP-', '-09-')
        dat = dat.replace('-OCT-', '-10-')
        dat = dat.replace('-NOV-', '-11-')
        dat = dat.replace('-DEC-', '-12-')
        self.date = datetime.strptime(dat, '%d-%m-%Y %H:%M:%S.%f')


        print 'Opened "{}", ({}x{})'.format(filename, self.width, self.height)

    def read_band(self, band_name, size, offset):
        '''
        offset: within the area of interest
        '''
        (ysize, xsize) = size
        (yoffset, xoffset) = offset
        return self.prod.get_band(band_name).read_as_array(
                    xoffset=xoffset, yoffset=yoffset+self.sline,
                    width=xsize, height=ysize)


    def read_bitmask(self, size, offset, bmexpr):

        (ysize, xsize) = size
        (yoffset, xoffset) = offset
        raster = epr.create_bitmask_raster(xsize, ysize)
        self.prod.read_bitmask_raster(bmexpr, xoffset, yoffset+self.sline, raster)

        return raster.data

    def read_block(self, size, offset, bands):
        '''
        size: size of the block
        offset: offset of the block
        bands: list of bands identifiers
        req: list of identifiers of required datasets
        '''

        (ysize, xsize) = size
        nbands = len(bands)

        # initialize block
        block = Block(offset=offset, size=size, bands=bands)

        block.latitude  = self.read_band('latitude',  size, offset)
        block.longitude = self.read_band('longitude', size, offset)

        # read geometry
        block.sza = self.read_band('sun_zenith', size, offset)
        block.vza = self.read_band('view_zenith', size, offset)
        block.saa = self.read_band('sun_azimuth', size, offset)
        block.vaa = self.read_band('view_azimuth', size, offset)

        # read detector index
        di = self.read_band('detector_index', size, offset)

        # get F0 for each band
        block.F0 = np.zeros((ysize, xsize, nbands)) + np.NaN
        for iband, band in enumerate(bands):
            block.F0[:,:,iband] = self.F0[self.F0_band_names[band]][di]

        # calculate detector wavelength for each band
        block.wavelen = np.zeros((ysize, xsize, nbands), dtype='float32') + np.NaN
        for iband, band in enumerate(bands):
            block.wavelen[:,:,iband] = self.detector_wavelength[self.wav_band_names[band]][di]

        # read TOA
        Ltoa = np.zeros((ysize,xsize,nbands)) + np.NaN
        for iband, band in enumerate(bands):
            Ltoa_ = self.read_band(self.band_names[band], size, offset)
            Ltoa[:,:,iband] = Ltoa_[:,:]
        block.Ltoa = Ltoa

        # wind speed (zonal and merdional)
        zwind = self.read_band('zonal_wind', size, offset)
        mwind = self.read_band('merid_wind', size, offset)
        block.wind_speed = np.sqrt(zwind**2 + mwind**2)

        # ozone
        block.ozone = self.read_band('ozone', size, offset)

        # surface pressure
        block.surf_press = self.read_band('atm_press', size, offset)

        # set julian day
        block.jday = self.date.timetuple().tm_yday

        # read bitmask
        block.bitmask = L2FLAGS['LAND']*self.read_bitmask(size, offset,
                'l1_flags.LAND_OCEAN').astype('uint16')
        block.bitmask += L2FLAGS['L1_INVALID']*self.read_bitmask(size, offset,
                '(l1_flags.INVALID) OR (l1_flags.SUSPECT) OR (l1_flags.COSMETIC)')

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

