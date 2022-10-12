#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from polymer.level2 import Level2
from polymer.level1_olci import Level1_OLCI
from polymer.main import run_atm_corr
from matplotlib import pyplot as plt
import numpy as np
from . import conftest



@pytest.mark.parametrize('uncertainties', [False, True])
@pytest.mark.parametrize('roi_desc,roi', [
    ('roi1_small', {
        'scol':1930+120,
        'ecol':2100,
        'sline':2556+40,
        'eline':2650,
    }),
    ('roi1', {
        'scol':1930,
        'ecol':2473,
        'sline':2556,
        'eline':3079,
    }),
    ('inland_water', {
        'scol':980,
        'ecol':1168,
        'sline':2524,
        'eline':2639,
    }),
])
def test_olci(request, uncertainties, roi_desc, roi):
    dsts = [
            'sza',
            'vza',
            'latitude',
            'longitude',
            'wind_speed',
            'ozone',
            'Rw',
            'Rtoa',
            ]
        
    if uncertainties:
        dsts += [
            'logchl_unc',
            'logfb_unc',
            'rho_w_unc',
            ]

    l2 = run_atm_corr(
        Level1_OLCI(
            '/rfs/user/francois/TESTCASES/OLCI/BALTIC/S3A_OL_1_EFR____20160720T093226_20160720T093526_20160720T113440_0180_006_307_1979_MAR_O_NR_001.SEN3/',
            **roi
        ),
        Level2('memory', datasets=dsts),
        bands_rw=[400,412,443,490,510,560,620,665,674,681,709,754,779,865,1020],
        uncertainties=uncertainties,
    )

    for dst in dsts:
        data = getattr(l2, dst)
        if dst == 'Rw':
            data = data[:,:,4]
            kw = {'vmin': 0, 'vmax': 0.05}
        elif dst == 'Rtoa':
            data = data[:,:,-1]
            kw = {'vmin': 0, 'vmax': 0.1}
        elif dst in ['logchl_unc', 'logfb_unc']:
            kw = {'vmin': min(0, np.nanpercentile(data, 5)),
                  'vmax': np.nanpercentile(data, 95)}
        elif dst in ['rho_w_unc']:
            data = data[:,:,0]
            kw = {'vmin': min(0, np.nanpercentile(data, 5)),
                  'vmax': np.nanpercentile(data, 95)}
        else:
            kw = {}

        plt.figure(figsize=(8,8))
        plt.imshow(data, **kw)
        plt.colorbar()
        plt.title(f'{dst} - {roi_desc}')
        conftest.savefig(request)


def test_spectrum(request):
    dsts = [
            'Rprime',
            'Rw',
            'Ratm',
            'Rtoa',
            'logchl_unc',
            'logfb_unc',
            'rho_w_unc',
            ]

    roi = {
        'scol':1930+120,
        'ecol':1930+120+1,
        'sline':2556+40,
        'eline':2556+40+1,
    }

    wav = [400,412,443,490,510,560,620,665,674,681,709,754,779,865,1020]
    l2 = run_atm_corr(
        Level1_OLCI(
            '/rfs/user/francois/TESTCASES/OLCI/BALTIC/S3A_OL_1_EFR____20160720T093226_20160720T093526_20160720T113440_0180_006_307_1979_MAR_O_NR_001.SEN3/',
            **roi
        ),
        Level2('memory', datasets=dsts),
        bands_rw=wav,
        uncertainties=True,
    )

    for var in ['Rprime', 'Ratm', 'Rw']:
        data = getattr(l2, var)
        plt.plot(wav, data[0,0,:], label=var)
    for var in [
            'logchl_unc',
            'logfb_unc',
            'rho_w_unc',
            ]:
        data = getattr(l2, var)
        print(var, data)

    plt.legend()
    plt.grid(True)
    conftest.savefig(request)
