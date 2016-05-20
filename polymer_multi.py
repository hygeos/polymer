#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
polymer multiprocessing version
'''

from multiprocessing import Pool
from polymer import convert_reflectance, Params_MERIS, Level1_MERIS, Level2_HDF
from polymer import gas_correction, cloudmask, rayleigh_correction
from polymer import PolymerMinimizer
from polymer import ParkRuddick as PR
from luts import read_mlut_hdf


def process_block(args):

    b, params, mlut, watermodel = args
    print 'process block', b

    convert_reflectance(b, params)

    gas_correction(b, params)

    cloudmask(b, params, mlut)

    rayleigh_correction(b, mlut, params)

    opt = PolymerMinimizer(watermodel.instantiate(), params)

    opt.minimize(b, params)

    return b

def blockiterator(params, blocks, mlut, watermodel):
    for b in blocks:
        yield (b, params, mlut, watermodel)


def polymer(params, level1, watermodel, level2):

    # initialize output file
    level2.init(level1)

    # read the look-up table
    mlut = read_mlut_hdf(params.lut_file)

    b_iter = level1.blocks(params.bands_read())
    blockiter = blockiterator(params, b_iter, mlut, watermodel)

    # process the blocks in parallel
    for b in Pool().imap_unordered(process_block, blockiter):
        assert 'Rtoa' in b.datasets()

        level2.write(b)

    level2.finish(params)

class ParkRuddick(object):
    '''
    A wrapper for ParkRuddick class
    (for compatibility with multiprocessing)

    Instantiates a Parkruddick class upon instantiate()
    '''
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
    def instantiate(self):
        return PR(*self.args, **self.kwargs)

