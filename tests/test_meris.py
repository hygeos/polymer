#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import pytest
from polymer.main import run_atm_corr
from polymer.level1_olci import Level1_OLCI
from polymer.level1_meris import Level1_MERIS
from polymer.level2 import Level2
from polymer.ancillary_era5 import Ancillary_ERA5
from matplotlib import pyplot as plt
from . import conftest





@pytest.mark.parametrize('processing', [3, 4])
@pytest.mark.parametrize('ancillary', [
    None,
    Ancillary_ERA5(),
])
@pytest.mark.parametrize('roi_desc,roi', [
    ('roi1', {'sline': 4426,
              'eline': 4834,
              'scol': 658,
              'ecol': 1079}),
    ('edge', {'sline': 0,
              'eline': 100,
              'scol': 0,
              'ecol': 100}),
    ('coast_inland', {'sline': 8000,
                      'eline': 8075,
                      'scol': 470,
                      'ecol': 650}),
])
def test_meris(request, ancillary, processing, roi, roi_desc):
    if processing == 3:
        product = '/rfs/user/francois/TESTCASES/MERIS_4TH/MER_RR__1PRACR20020503_105146_000026382005_00352_00907_0000.N1'
    else:
        product = '/rfs/user/francois/TESTCASES/MERIS_4TH/ENV_ME_1_RRG____20020503T105146_20020503T113544_________________2638_005_352______DSI_R_NT____.SEN3/'

    l1 = Level1_MERIS(
        product,
        ancillary=ancillary,
        **roi,
    )

    dsts = [
            'sza',
            'vza',
            # 'latitude',
            # 'longitude',
            'wind_speed',
            'ozone',
            'Rw',
            'Rtoa',
            ]
    l2 = run_atm_corr(l1, Level2('memory', datasets=dsts))
    for dst in dsts:
        data = getattr(l2, dst)
        if dst == 'Rw':
            data = data[:,:,4]
        elif dst == 'Rtoa':
            data = data[:,:,-1]

        plt.figure(figsize=(8,8))
        plt.imshow(data)
        plt.colorbar()
        plt.title(f'{dst} - meris_processing={processing} - {roi_desc}')
        conftest.savefig(request)
