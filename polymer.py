#!/usr/bin/env python
# encoding: utf-8


from level1_meris import Level1_MERIS
from level2 import Level2_Memory, RGB, Level2_HDF
import numpy as np
from luts import read_mlut_hdf, LUT, Idx
from pylab import imshow, show, colorbar
import warnings
from utils import stdNxN
from common import BITMASK_INVALID

# cython imports
# import pyximport ; pyximport.install()
from polymer_main import PolymerMinimizer
from water import ParkRuddick


'''
quelques notes pour le développement éventuel de Polymer en python
'''

# TODO
# renommer ths, thv => sza, sza_deg, etc
# paramètres spécifiques aux capteurs: à passer aux objets L1
# au moins bands_L1, etc


class Params(object):
    def __init__(self):
        '''
        define common parameters
        '''
        pass


    def bands_read(self):
        bands_read = set(self.bands_corr)
        bands_read = bands_read.union(self.bands_oc)
        bands_read = bands_read.union(self.bands_rw)
        return sorted(bands_read)


class Params_MERIS(Params):
    def __init__(self):
        super(self.__class__, self).__init__()

        self.bands_corr = [412,443,490,510,560,620,665,        754,    779,865]
        self.bands_oc =   [412,443,490,510,560,620,665,        754,    779,865]
        self.bands_rw =   [412,443,490,510,560,620,665,        754,    779,865]

        self.lut_file = '/home/francois/MERIS/POLYMER/LUTS/MERIS/LUTB.hdf'
        self.lut_bands = [412,443,490,510,560,620,665,681,709,754,760,779,865,885,900]

        self.band_cloudmask = 865
        self.thres_Rcloud = 0.2
        self.thres_Rcloud_std = 0.04


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

    for i in xrange(block.nbands):

        block.Rtoa[i,:,:] = block.Ltoa[i,:,:]*np.pi/(block.mus*block.F0[i,:,:]*coef)


def gas_correction(block, l1):

    block.Rtoa_gc = np.zeros(block.Rtoa.shape, dtype='float32') + np.NaN

    # ok = (block.bitmask & BITMASK_INVALID) == 0
    # FIXME

    #
    # ozone correction
    #
    # make sure that ozone is in DU
    if (block.ozone < 50).any() or (block.ozone > 1000).any():
        raise Exception('Error, ozone is assumed in DU')

    # bands loop
    for i, b in enumerate(block.bands):

        tauO3 = l1.K_OZ[b] * block.ozone * 1e-3  # convert from DU to cm*atm

        # ozone transmittance
        trans_O3 = np.exp(-tauO3 * block.air_mass)

        block.Rtoa_gc[i,:,:] = block.Rtoa[i,:,:]/trans_O3

    #
    # NO2 correction
    #
    warnings.warn('TODO: implement NO2 correction')


def cloudmask(b, params, mlut):

    # FIXME
    # cloud mask is not properly calculated at the edges of blocks

    inir_block = b.bands.index(params.band_cloudmask)
    inir_lut = params.lut_bands.index(params.band_cloudmask)
    b.Rnir = b.Rtoa_gc[inir_block,:,:] - mlut['Rmol'][
            Idx(b.muv),
            Idx(b.raa),
            Idx(b.mus),
            inir_lut]
    b.cloudmask = b.Rnir > params.thres_Rcloud
    b.cloudmask |= stdNxN(b.Rnir, 3) > params.thres_Rcloud_std


def rayleigh_correction(block, mlut, params):
    '''
    Rayleigh correction
    + transmission interpolation
    '''

    block.Rprime = np.zeros(block.Ltoa.shape, dtype='float32')+np.NaN
    block.Tmol = np.zeros(block.Ltoa.shape, dtype='float32')+np.NaN

    # TODO: don't process masked

    for i in xrange(block.nbands):
        ilut = params.lut_bands.index(block.bands[i])

        block.Rprime[i,:,:] = block.Rtoa_gc[i,:,:] - mlut['Rmolgli'][
                Idx(block.muv),
                Idx(block.raa),
                Idx(block.mus),
                ilut, Idx(block.wind_speed)]

        # TODO: share axes indices
        block.Tmol[i,:,:]  = mlut['Tmolgli'][Idx(block.mus), ilut, Idx(block.wind_speed)]
        block.Tmol[i,:,:] *= mlut['Tmolgli'][Idx(block.muv), ilut, Idx(block.wind_speed)]


def polymer(params, level1, watermodel, level2):
    '''
    Polymer atmospheric correction
    '''

    # initialize output file
    level2.init(level1)

    # read the look-up table
    mlut = read_mlut_hdf(params.lut_file)

    opt = PolymerMinimizer(watermodel)

    # loop over the blocks
    for b in level1.blocks(params.bands_read()):

        convert_reflectance(b, params)

        gas_correction(b, level1)

        cloudmask(b, params, mlut)

        rayleigh_correction(b, mlut, params)

        # FIXME
        # warnings.warn('TODO: reactivate minimization')
        # opt.minimize(b, params)

        level2.write(b)

    level2.finish()

    return level2

