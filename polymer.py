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
from polymer_main import polymer_optimize


'''
quelques notes pour le développement éventuel de Polymer en python
'''

# TODO
# renommer ths, thv => sza, sza_deg, etc
# paramètres spécifiques aux capteurs: à passer aux objets L1
# au moins bands_L1, etc


class Params(object):
    def bands_read(self):
        bands_read = set(self.bands_corr)
        bands_read = bands_read.union(self.bands_oc)
        bands_read = bands_read.union(self.bands_rw)
        return sorted(bands_read)


class Params_MERIS(Params):
    def __init__(self):
        self.bands_l1   = [412,443,490,510,560,620,665,681,709,754,760,779,865,885,900]
        self.bands_corr = [412,443,490,510,560,620,665,        754,    779,865]
        self.bands_oc =   [412,443,490,510,560,620,665,        754,    779,865]
        self.bands_rw =   [412,443,490,510,560,620,665,        754,    779,865]
        self.lut_file = '/home/francois/MERIS/POLYMER/LUTS/MERIS/LUTB.hdf'

        # read solar irradiance

def convert_reflectance(block, params):

    block.set('Rtoa', np.zeros(block.Ltoa.shape)+np.NaN)

    coef = 1.   # FIXME, sun-earth distance coefficient

    for i, b in enumerate(block.bands):

        mus = np.cos(block.ths*np.pi/180.)

        block.Rtoa[i,:,:] = block.Ltoa[i,:,:]*np.pi/(mus*block.F0[i,:,:]*coef)

def init_block(block):
    # TODO: move this to class block ?

    if not hasattr(block, 'mus'):
        setattr(block, 'mus', np.cos(block.ths*np.pi/180.))

    if not hasattr(block, 'muv'):
        setattr(block, 'muv', np.cos(block.thv*np.pi/180.))

    # calculate relative attribute if necessary
    if not hasattr(block, 'phi'):
        phi = block.phis-block.phiv
        phi[phi<0.] += 360;
        phi[phi>360.] -= 360;
        phi[phi>180.] = 360. - phi[phi>180.]
        block.set('phi', phi)

    # FIXME
    block.set('wind', np.zeros_like(block.phi)+5.)


def gas_correction(block):
    pass

def rayleigh_correction(block, mlut):

    block.set('Rprime', np.zeros(block.Ltoa.shape)+np.NaN)

    for i, b in enumerate(block.bands):

        block.Rprime[i,:,:] = block.Rtoa[i,:,:] - mlut['Rmolgli'][
                Idx(block.muv),
                Idx(block.phi),
                Idx(block.mus),
                i, Idx(block.wind)]
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

    # loop over the blocks
    for i, b in enumerate(level1.blocks(params.bands_read())):

        init_block(b)

        convert_reflectance(b, params)

        gas_correction(b)

        rayleigh_correction(b, mlut)

        polymer_optimize(b)

        level2.write(b)

    level2.finish()

def main():
    l2 = Level2_Memory()
    polymer(
            Params_MERIS(),
            Level1_MERIS('/mfs/proj/CNES_GLITTER_2009/DATA_HYGEOS/20041104_06/MER_RR__1PQBCM20041105_060121_000002002031_00449_14030_0002.N1'),
            l2,
            )
    # RGB(l2.Rtoa)
    # RGB(l2.Rprime)
    # show()

if __name__ == '__main__':
    main()

