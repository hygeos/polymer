#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import basename


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

        elif b.startswith('S2A_OPER_MSI_L1C'):
            self.sensor = 'msi'

        else:
            raise Exception('Unable to detect sensor for file "{}"'.format(b))

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







