#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from os.path import basename, join
import numpy as np
from itertools import product
from glob import glob



class Level1(object):
    '''
    Level 1 initializer
    Creates a Level1_* instance
    If sensor is not provided, auto-detects the sensor
    based on file name.
    NOTE: allows to instanciater the Level1* object in
    the 'with' block

    ARGUMENTS:
    filename: path to level 1
    sensor: sensor name
    other kwargs are passed to the Level1_* constructor
    '''

    def __init__(self, filename, sensor=None, **kwargs):

        self.sensor = sensor
        self.filename = filename
        self.basename = basename(filename)
        self.kwargs = kwargs
        self.level1 = None

        if sensor is None:
            self.autodetect()

    def autodetect(self):

        b = self.basename

        if (b.startswith('MER_RR') or b.startswith('MER_FR')) and b.endswith('.N1'):
            self.sensor = 'meris'

        elif b.startswith('S3A_OL_1') and b.endswith('.SEN3'):
            self.sensor = 'olci'

        elif b.startswith('V') and '.L1C' in b:
            self.sensor = 'viirs'

        elif b.startswith('A') and '.L1C' in b:
            self.sensor = 'modis'

        elif b.startswith('S') and '.L1C' in b:
            self.sensor = 'seawifs'

        elif self.detect_msi():
            self.sensor = 'msi'

        else:
            raise Exception('Unable to detect sensor for file "{}"'.format(b))


    def detect_msi(self):
        xmlfiles = glob(join(self.filename, '*MTD*_TL*.xml'))
        return len(xmlfiles) == 1


    def __str__(self):
        return '<{} level1: {}>'.format(self.sensor, self.basename)

    def __enter__(self):
        '''
        Instantiate the level1 object
        (in a 'with' context)
        '''

        assert self.level1 is None
        if self.sensor == 'meris':
            from polymer.level1_meris import Level1_MERIS
            L1 = Level1_MERIS

        elif self.sensor == 'olci':
            from polymer.level1_olci import Level1_OLCI
            L1 = Level1_OLCI

        elif self.sensor == 'viirs':
            from polymer.level1_nasa import Level1_VIIRS
            L1 = Level1_VIIRS

        elif self.sensor == 'modis':
            from polymer.level1_nasa import Level1_MODIS
            L1 = Level1_MODIS

        elif self.sensor == 'seawifs':
            from polymer.level1_nasa import Level1_SeaWiFS
            L1 = Level1_SeaWiFS

        elif self.sensor == 'msi':
            from polymer.level1_msi import Level1_MSI
            L1 = Level1_MSI

        else:
            raise Exception('Invalid sensor name "{}"'.format(self.sensor))

        self.level1 = L1(self.filename, **self.kwargs)
        return self.level1

    def __exit__(self, *args):
        self.level1 = None



class Level1_base(object):
    '''
    Base class for Level1 objects
    '''

    def init_shape(self, totalheight, totalwidth,
                   sline=0, eline=-1,
                   scol=0, ecol=-1):
        self.totalheight = totalheight
        self.totalwidth = totalwidth

        if sline > totalheight:
            raise IndexError('Invalid sline {} (product height is {})'.format(sline, totalheight))
        if scol > totalwidth:
            raise IndexError('Invalid scol {} (product width is {})'.format(scol, totalwidth))

        self.sline = sline
        self.eline = eline
        self.scol = scol
        self.ecol = ecol

        if eline < 0:
            self.height = self.totalheight
            self.height -= sline
            self.height += eline + 1
        else:
            self.height = eline-sline

        if ecol < 0:
            self.width = self.totalwidth
            self.width -= scol
            self.width += ecol + 1
        else:
            self.width = ecol - scol

        self.shape = (self.height, self.width)


    def blocks(self, bands_read):

        nblocks_h = int(np.ceil(float(self.height)/self.blocksize[0]))
        nblocks_w = int(np.ceil(float(self.width)/self.blocksize[1]))

        for (iblock_h, iblock_w) in product(range(nblocks_h), range(nblocks_w)):

            # determine block size
            if iblock_h == nblocks_h-1:
                ysize = self.height-(nblocks_h-1)*self.blocksize[0]
            else:
                ysize = self.blocksize[0]
            if iblock_w == nblocks_w-1:
                xsize = self.width-(nblocks_w-1)*self.blocksize[1]
            else:
                xsize = self.blocksize[1]
            size = (ysize, xsize)

            # determine the block offset
            yoffset = iblock_h * self.blocksize[0]
            xoffset = iblock_w * self.blocksize[1]
            offset = (yoffset, xoffset)

            yield self.read_block(size, offset, bands_read)


