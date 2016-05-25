#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
polymer multiprocessing version
'''

from multiprocessing import Pool
from polymer import InitCorr


def process_block(args):

    b, c = args
    print 'process block', b

    c.convert_reflectance(b)

    c.gas_correction(b)

    c.cloudmask(b)

    c.rayleigh_correction(b)

    opt = c.init_minimizer()

    opt.minimize(b, c.params)

    return b

def blockiterator(blocks, initcorr):
    for b in blocks:
        yield (b, initcorr)


def polymer(level1, params, level2):

    # initialize output file
    level2.init(level1)

    c = InitCorr(params)

    b_iter = level1.blocks(params.bands_read())
    blockiter = blockiterator(b_iter, c)

    # process the blocks in parallel
    for b in Pool().imap_unordered(process_block, blockiter):
        assert 'Rtoa' in b.datasets()

        level2.write(b)

    level2.finish(params)

