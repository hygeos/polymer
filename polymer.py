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


'''
quelques notes pour le développement éventuel de Polymer en python
'''

# TODO
# paramètres spécifiques aux capteurs: à passer aux objets L1
# au moins bands_L1, etc


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

        block.Rtoa[i,ok] = block.Ltoa[i,ok]*np.pi/(block.mus[ok]*block.F0[i,ok]*coef)


def gas_correction(block, l1):

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

        tauO3 = l1.K_OZ[b] * block.ozone[ok] * 1e-3  # convert from DU to cm*atm

        # ozone transmittance
        trans_O3 = np.exp(-tauO3 * block.air_mass[ok])

        block.Rtoa_gc[i,ok] = block.Rtoa[i,ok]/trans_O3

    #
    # NO2 correction
    #
    warnings.warn('TODO: implement NO2 correction')


def cloudmask(block, params, mlut):

    ok = (block.bitmask & BITMASK_INVALID) == 0

    inir_block = block.bands.index(params.band_cloudmask)
    inir_lut = params.lut_bands.index(params.band_cloudmask)
    block.Rnir = block.Rtoa_gc[inir_block,:,:] - mlut['Rmol'][
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

        block.Rprime[i,ok] = block.Rtoa_gc[i,ok] - mlut['Rmolgli'][
                Idx(block.muv[ok]),
                Idx(block.raa[ok]),
                Idx(block.mus[ok]),
                ilut, Idx(block.wind_speed[ok])]

        # TODO: share axes indices
        block.Tmol[i,ok]  = mlut['Tmolgli'][Idx(block.mus[ok]),
                ilut, Idx(block.wind_speed[ok])]
        block.Tmol[i,ok] *= mlut['Tmolgli'][Idx(block.muv[ok]),
                ilut, Idx(block.wind_speed[ok])]


def polymer(params, level1, watermodel, level2):
    '''
    Polymer atmospheric correction
    '''

    # initialize output file
    level2.init(level1)

    # read the look-up table
    mlut = read_mlut_hdf(params.lut_file)

    opt = PolymerMinimizer(watermodel, params)

    # loop over the blocks
    for b in level1.blocks(params.bands_read()):

        convert_reflectance(b, params)

        gas_correction(b, level1)

        cloudmask(b, params, mlut)

        rayleigh_correction(b, mlut, params)

        opt.minimize(b, params)

        level2.write(b)

    level2.finish(params)

    return level2

