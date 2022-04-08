#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .level1_safe import Level1_SAFE

central_wavelength_olci = {
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


def Level1_OLCI(dirname,
                sline=0, eline=-1,
                scol=0, ecol=-1,
                blocksize=100, ancillary=None,
                landmask='default',
                altitude=0.,
                add_noise=False,
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

    central_wavelength = central_wavelength_olci

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
    

    sigma_typ_olci = {
        400: 0.079123, 412: 0.075843, 443: 0.066794, 490: 0.049074,
        510: 0.043102, 560: 0.033475, 620: 0.028571, 665: 0.021893,
        674: 0.024714, 681: 0.0282, 709: 0.019318, 754: 0.018199,
        760: 0.033306, 764: 0.036202, 767: 0.035583, 779: 0.011097,
        865: 0.010115, 885: 0.01282, 900: 0.016216, 940: 0.031001,
        1020:0.085567,
    }

    Ltyp_olci = {
        400:85.583336, 412:81., 443:70., 490:53., 510:45.3,
        560:31.4, 620:20.4, 665:15.5, 674:14.746154, 681:14.1,
        709:11.9, 754:9.5, 760:7.469231, 764:7.477778,
        767:7.7, 779:8.5, 865:5.3, 885:4.750141,
        900:4.2, 940:2.732958, 1020:2.267843,
    }

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
        Ltyp=Ltyp_olci,
        sigma_typ=sigma_typ_olci,
        add_noise=add_noise,
    )
