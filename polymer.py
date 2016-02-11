#!/usr/bin/env python
# encoding: utf-8


from level1_meris import Level1_MERIS
from level2_hdf import Level2_HDF
from level2_view import Level2_Memory, RGB
import numpy as np
from luts import read_mlut_hdf, LUT, Idx
from pylab import imshow, show, colorbar
import warnings

# cython imports
import pyximport ; pyximport.install()
from polymer_main import PolymerMinimizer


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

        # read solar irradiance

def convert_reflectance(block, params):

    block.Rtoa = np.zeros(block.Ltoa.shape)+np.NaN

    coef = 1.   # FIXME, sun-earth distance coefficient

    for i in xrange(block.nbands):

        block.Rtoa[i,:,:] = block.Ltoa[i,:,:]*np.pi/(block.mus*block.F0[i,:,:]*coef)


def gas_correction(block):
    pass

def rayleigh_correction(block, mlut):

    block.Rprime = np.zeros(block.Ltoa.shape, dtype='float32')+np.NaN

    for i in xrange(block.nbands):

        block.Rprime[i,:,:] = block.Rtoa[i,:,:] - mlut['Rmolgli'][
                Idx(block.muv),
                Idx(block.raa),
                Idx(block.mus),
                i, Idx(block.wind_speed)]
    # print block.Rprime.data[:,:,:]
        # FIXME:
        # ordre mus muv
        # fournir la vitesse du vent
        # clarifier les indices des bandes


def polymer(params, level1, level2):

    # initialize output file
    level2.init(level1)

    # read the look-up table
    mlut = read_mlut_hdf(params.lut_file)

    opt = PolymerMinimizer()

    # loop over the blocks
    for b in level1.blocks(params.bands_read()):

        convert_reflectance(b, params)

        gas_correction(b)

        rayleigh_correction(b, mlut)

        opt.minimize(b)

        level2.write(b)

    level2.finish()

    return level2

def main():
    # l2 = Level2_Memory()
    l2 = polymer(
            Params_MERIS(),
            Level1_MERIS('/mfs/proj/CNES_GLITTER_2009/DATA_HYGEOS/20041104_06/MER_RR__1PQBCM20041105_060121_000002002031_00449_14030_0002.N1'),
            Level2_Memory(),
            )
    # RGB(LUT(l2.Rtoa, axes=[l2.bands, None, None]))
    # RGB(LUT(l2.Rprime, axes=[l2.bands, None, None]))
    # show()

if __name__ == '__main__':
    main()

