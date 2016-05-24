#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
polymer multiprocessing version
'''

from multiprocessing import Pool
from polymer import convert_reflectance
from polymer import gas_correction, cloudmask, rayleigh_correction
from polymer import PolymerMinimizer, init_water_model
from luts import read_mlut_hdf


def process_block(args):

    b, params, mlut = args
    print 'process block', b

    convert_reflectance(b, params)

    gas_correction(b, params)

    cloudmask(b, params, mlut)

    rayleigh_correction(b, mlut, params)

    watermodel = init_water_model(params)

    opt = PolymerMinimizer(watermodel, params)

    opt.minimize(b, params)

    return b

def blockiterator(params, blocks, mlut):
    for b in blocks:
        yield (b, params, mlut)


def polymer(level1, params, level2):

    # initialize output file
    level2.init(level1)

    # read the look-up table
    mlut = read_mlut_hdf(params.lut_file)

    b_iter = level1.blocks(params.bands_read())
    blockiter = blockiterator(params, b_iter, mlut)

    # process the blocks in parallel
    for b in Pool().imap_unordered(process_block, blockiter):
        assert 'Rtoa' in b.datasets()

        level2.write(b)

    level2.finish(params)

