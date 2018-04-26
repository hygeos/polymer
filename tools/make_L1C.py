#!/usr/bin/env python
# vim:fileencoding=utf-8


'''
A small script to produce GEO, L1B and L1C files for MODIS and VIIRS
(which include top of atmosphere radiances and polarization correction)
'''

from __future__ import print_function
import os
from sys import exit, argv
from os.path import exists, basename, join, isdir
from tmpfiles import TmpOutput
from glob import glob


def process(l1a):
    if l1a.endswith('/'):
        l1a = l1a[:-1]
    if basename(l1a).startswith('V'):
        if isdir(l1a):
            process_viirs_legacy(l1a)
        else:
            process_viirs(l1a)
    elif basename(l1a).startswith('S'):
        process_seawifs(l1a)
    elif basename(l1a).startswith('A'):
        process_modis(l1a)
    else:
        print('Invalid processor for', l1a)


def make_L1C(ifile, filename_geo, filename_l1c, nbands):
    '''
    make level1c using l2gen for MODIS and VIIRS
    '''

    if not exists(filename_l1c):
        with TmpOutput(filename_l1c) as f:
            gains = ' '.join(['1.0']*nbands)
            cmd = 'l2gen ifile="{}" geofile="{}" ofile="{}" ' \
                  'l2prod="rhot_nnn polcor_nnn sena senz sola solz latitude longitude" ' \
                  'gain="{}" atmocor=0 aer_opt=-99 brdf_opt=0'.format(
                          ifile, filename_geo, f, gains)
            if os.system(cmd):
                print('exiting l2gen')
                exit(1)
            f.move()
    else:
        print('Skipping existing', filename_l1c)


def process_modis(filename_l1a):
    '''
    generate MODIS level 1c from level 1a
    '''

    if not filename_l1a.endswith('.L1A_LAC'):
        print('Skipping %s' % (filename_l1a))
        return

    filename_geo = filename_l1a[:-8] + '.GEO'
    filename_l1b = filename_l1a[:-8] + '.L1B_LAC'
    filename_l1c = filename_l1a[:-8] + '.L1C'

    print('  -> processing', filename_l1a)

    #
    # GEO file
    #
    if not exists(filename_geo):
        with TmpOutput(filename_geo) as f:
            cmd = 'modis_GEO.py --output={} {}'.format(f, filename_l1a)
            print(cmd)
            if os.system(cmd):
                print('Error in modis_GEO')
                exit(1)
            f.move()
    else:
        print('Skipping existing', filename_geo)

    #
    # L1B
    #
    if not exists(filename_l1b):
        with TmpOutput(filename_l1b) as f:
            if os.system('modis_L1B.py -y -z --okm={} {} {}'.format(f, filename_l1a, filename_geo)):
                print('Error in modis_l1B')
                exit(1)
            f.move()
    else:
        print('Skipping existing', filename_l1b)

    #
    # L1C
    #
    make_L1C(filename_l1b, filename_geo, filename_l1c, 16)


def process_viirs_legacy(directory_l1a):
    '''
    generate VIIRS level 1c from level 1a (a directory)
    '''

    filename_l1a = glob(join(directory_l1a, 'SVM01*.h5'))[0]
    filename_geo = glob(join(directory_l1a, 'GMTCO*.h5'))[0]
    if directory_l1a.endswith('/'):
        directory_l1a = directory_l1a[:-1]
    filename_l1c = directory_l1a.replace('.L1A_NPP', '.L1C')

    make_L1C(filename_l1a, filename_geo, filename_l1c, 10)


def process_viirs(filename_l1a):
    '''
    generate VIIRS level 1c from level 1a
    (netcdf format)
    '''

    assert '.nc' in filename_l1a
    filename_geo = filename_l1a.replace('.L1A_SNPP.nc', '.GEO-M_SNPP.nc')
    filename_l1c = filename_l1a.replace('.L1A_SNPP.nc', '.L1C')

    #
    # GEO file
    #
    if not exists(filename_geo):
        with TmpOutput(filename_geo) as f:
            if os.system('geolocate_viirs ifile={} geofile_mod={}'.format(filename_l1a, f)):
                print('Error in modis_GEO')
                exit(1)
            f.move()
    else:
        print('Skipping existing', filename_geo)

    make_L1C(filename_l1a, filename_geo, filename_l1c, 10)


def process_seawifs(filename_l1a):
    filename_l1c = filename_l1a.replace('.L1A', '.L1C')
    if not exists(filename_l1c):
        with TmpOutput(filename_l1c) as f:
            cmd = 'l2gen ifile={} ofile={} gain="1 1 1 1 1 1 1 1" oformat="netcdf4" l2prod="rhot_nnn polcor_nnn sena senz sola solz latitude longitude"'.format(filename_l1a, f)
            if os.system(cmd):
                print('exiting l2gen')
                exit(1)
            f.move()
    else:
        print('Skipping existing', filename_l1c)


if __name__ == '__main__':

    for l1a in argv[1:]:
        process(l1a)
