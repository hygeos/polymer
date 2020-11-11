#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .level1_safe import Level1_SAFE


def Level1_OLCI(dirname,
                sline=0, eline=-1,
                scol=0, ecol=-1,
                blocksize=100, ancillary=None,
                landmask='default',
                altitude=0.,
                ):
    '''
    OLCI reader (SAFE format)

    landmask:
        * None => don't apply land mask at all
        * 'default' => use landmask provided in Level1
        * GSW object: use global surface water product (see gsw.py)

    altitude: surface altitude in m
        * a float
        * a DEM instance such as:
            SRTM3(cache_dir=...)  # srtm.py
            GLOBE(directory=...)  # globe.py
            SRTM3(..., missing=GLOBE(...))
    '''
    # central wavelength of the detector (for normalization)
    # (detector 374 of camera 3)
    central_wavelength = {
            400 : 400.664  , 412 : 412.076 ,
            443 : 443.183  , 490 : 490.713 ,
            510 : 510.639  , 560 : 560.579 ,
            620 : 620.632  , 665 : 665.3719,
            674 : 674.105  , 681 : 681.66  ,
            709 : 709.1799 , 754 : 754.2236,
            760 : 761.8164 , 764 : 764.9075,
            767 : 767.9734 , 779 : 779.2685,
            865 : 865.4625 , 885 : 884.3256,
            900 : 899.3162 , 940 : 939.02  ,
            1020: 1015.9766, 1375: 1375.   ,
            1610: 1610.    , 2250: 2250.   ,
            }

    band_names = {
        400 : 'Oa01_radiance', 412 : 'Oa02_radiance',
        443 : 'Oa03_radiance', 490 : 'Oa04_radiance',
        510 : 'Oa05_radiance', 560 : 'Oa06_radiance',
        620 : 'Oa07_radiance', 665 : 'Oa08_radiance',
        674 : 'Oa09_radiance', 681 : 'Oa10_radiance',
        709 : 'Oa11_radiance', 754 : 'Oa12_radiance',
        760 : 'Oa13_radiance', 764 : 'Oa14_radiance',
        767 : 'Oa15_radiance', 779 : 'Oa16_radiance',
        865 : 'Oa17_radiance', 885 : 'Oa18_radiance',
        900 : 'Oa19_radiance', 940 : 'Oa20_radiance',
        1020: 'Oa21_radiance',
        }

    band_index = {
        400 : 0, 412: 1, 443 : 2, 490: 3,
        510 : 4, 560: 5, 620 : 6, 665: 7,
        674 : 8, 681: 9, 709 :10, 754: 11,
        760 :12, 764: 13, 767 :14, 779: 15,
        865 :16, 885: 17, 900 :18, 940: 19,
        1020:20}

    return Level1_SAFE(
        dirname,
        sline=sline, eline=eline,
        scol=scol, ecol=ecol,
        blocksize=blocksize,
        ancillary=ancillary,
        landmask=landmask,
        altitude=altitude,
        sensor='OLCI',
        central_wavelength=central_wavelength,
        band_names=band_names,
        band_index=band_index,
    )
