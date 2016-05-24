#!/usr/bin/env python
# encoding: utf-8


from level1_meris import Level1_MERIS
from level2 import Level2_Memory, RGB, Level2_HDF
import numpy as np
from luts import read_mlut_hdf, LUT, Idx
from pylab import imshow, show, colorbar
import warnings
from utils import stdNxN
from common import BITMASK_INVALID, L2FLAGS
from collections import OrderedDict

# cython imports
# import pyximport ; pyximport.install()
from polymer_main import PolymerMinimizer
from water import ParkRuddick


class Params(object):
    '''
    Sensor non-specific parameters
    '''
    def __init__(self):
        '''
        define common parameters
        '''

        # cloud masking
        self.thres_Rcloud = 0.2
        self.thres_Rcloud_std = 0.04

        # optimization parameters
        self.force_initialization = False
        self.max_iter = 100
        self.size_end_iter = 0.005
        self.initial_point = [-1, 0]
        self.initial_step = [0.2, 0.2]
        self.bounds = [[-2, 2], [-3, 3]]

        self.thres_chi2 = 0.005

        # Constraint on bbs: amplitude, sigma(chl=0.01), sigma(chl=0.1)
        # (disactivate with amplitude == 0)
        self.constraint_bbs = [1e-3, 0.2258, 0.9233]

        self.partial = 0    # whether to perform partial processing
                            #       0: standard processing
                            #       1: stop at minimize
                            #       2: stop at rayleigh correction

    def bands_read(self):
        bands_read = set(self.bands_corr)
        bands_read = bands_read.union(self.bands_oc)
        bands_read = bands_read.union(self.bands_rw)
        return sorted(bands_read)

    def print_info(self):
        print self.__class__
        for k, v in self.__dict__.iteritems():
            print '*', k,':', v

    def update(self, **kwargs):

        # don't allow for 'new' attributes
        for k in kwargs:
            if k not in self.__dict__:
                raise Exception('{}: attribute "{}" is unknown'.format(self.__class__, k))

        self.__dict__.update(kwargs)

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
        self.lut_bands = [412,443,490,510,560,620,665,681,709,754,760,779,865,885,900]

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
        self.NO2_CLIMATOLOGY = '/home/francois/MERIS/POLYMER/auxdata/common/no2_climatology.hdf'
        self.NO2_FRAC200M = '/home/francois/MERIS/POLYMER/auxdata/common/trop_f_no2_200m.hdf'

        # update 
        self.update(**kwargs)


def coeff_sun_earth_distance(jday):
    A=1.00014
    B=0.01671
    C=0.9856002831
    D=3.4532858
    E=360.
    F=0.00014

    coef  = 1./((A - B*np.cos(2*np.pi*(C*jday - D)/E) - F*np.cos(4*np.pi*(C*jday - D)/E))**2)

    return coef

def convert_reflectance(block, params):

    block.Rtoa = np.zeros(block.Ltoa.shape)+np.NaN

    coef = coeff_sun_earth_distance(block.jday)

    ok = (block.bitmask & BITMASK_INVALID) == 0

    for i in xrange(block.nbands):

        block.Rtoa[ok,i] = block.Ltoa[ok,i]*np.pi/(block.mus[ok]*block.F0[ok,i]*coef)


def gas_correction(block, params):

    block.Rtoa_gc = np.zeros(block.Rtoa.shape, dtype='float32') + np.NaN

    ok = (block.bitmask & BITMASK_INVALID) == 0

    #
    # ozone correction
    #
    # make sure that ozone is in DU
    if (block.ozone < 50).any() or (block.ozone > 1000).any():
        raise Exception('Error, ozone is assumed in DU')

    # bands loop
    for i, b in enumerate(block.bands):

        tauO3 = params.K_OZ[b] * block.ozone[ok] * 1e-3  # convert from DU to cm*atm

        # ozone transmittance
        trans_O3 = np.exp(-tauO3 * block.air_mass[ok])

        block.Rtoa_gc[ok,i] = block.Rtoa[ok,i]/trans_O3

    #
    # NO2 correction
    #
    warnings.warn('TODO: implement NO2 correction')


def cloudmask(block, params, mlut):
    '''
    Polymer basic cloud mask
    '''

    ok = (block.bitmask & BITMASK_INVALID) == 0

    inir_block = block.bands.index(params.band_cloudmask)
    inir_lut = params.lut_bands.index(params.band_cloudmask)
    block.Rnir = block.Rtoa_gc[:,:,inir_block] - mlut['Rmol'][
            Idx(block.muv),
            Idx(block.raa),
            Idx(block.mus),
            inir_lut]
    cloudmask = block.Rnir > params.thres_Rcloud
    cloudmask |= stdNxN(block.Rnir, 3, ok) > params.thres_Rcloud_std

    block.bitmask += L2FLAGS['CLOUD_BASE'] * cloudmask.astype('uint8')


def rayleigh_correction(block, mlut, params):
    '''
    Rayleigh correction
    + transmission interpolation
    '''
    if params.partial >= 2:
        return

    block.Rprime = np.zeros(block.Ltoa.shape, dtype='float32')+np.NaN
    block.Tmol = np.zeros(block.Ltoa.shape, dtype='float32')+np.NaN

    ok = (block.bitmask & BITMASK_INVALID) == 0

    for i in xrange(block.nbands):
        ilut = params.lut_bands.index(block.bands[i])

        block.Rprime[ok,i] = block.Rtoa_gc[ok,i] - mlut['Rmolgli'][
                Idx(block.muv[ok]),
                Idx(block.raa[ok]),
                Idx(block.mus[ok]),
                ilut, Idx(block.wind_speed[ok])]

        # TODO: share axes indices
        block.Tmol[ok,i]  = mlut['Tmolgli'][Idx(block.mus[ok]),
                ilut, Idx(block.wind_speed[ok])]
        block.Tmol[ok,i] *= mlut['Tmolgli'][Idx(block.muv[ok]),
                ilut, Idx(block.wind_speed[ok])]


def init_water_model(params):

    return ParkRuddick('/home/francois/MERIS/POLYMER/auxdata/common/')


def polymer(level1, params, level2):
    '''
    Polymer atmospheric correction
    '''

    # initialize output file
    level2.init(level1)

    # read the look-up table
    mlut = read_mlut_hdf(params.lut_file)

    watermodel = init_water_model(params)

    opt = PolymerMinimizer(watermodel, params)

    # loop over the blocks
    for b in level1.blocks(params.bands_read()):

        convert_reflectance(b, params)

        gas_correction(b, params)

        cloudmask(b, params, mlut)

        rayleigh_correction(b, mlut, params)

        opt.minimize(b, params)

        level2.write(b)

    level2.finish(params)

    return level2

