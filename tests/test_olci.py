#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from core import pytest_utils
from core.env import getdir
from core.fileutils import mdir
from eoread import autodetect, olci
from eoread.common import timeit
from matplotlib import pyplot as plt

from polymer.level1 import Level1
from polymer.level1_olci import Level1_OLCI
from polymer.level2 import Level2, OutputExists
from polymer.level2_nc import Level2_NETCDF
from polymer.main import run_atm_corr
from polymer.main_v5 import run_polymer, run_polymer_dataset
from tests.common import diff, diff_flags, load_polymer, plot, run_v4, run_v5

from . import conftest

olci_level1 = str(
    getdir('DIR_DATA')/'sample_products'/
    'S3A_OL_1_EFR____20160720T093221_20160720T093421_20171002T063740_0119_006_307______MR1_R_NT_002.SEN3')


@pytest.fixture
def testcase():
    return {
        # Gironde estuary
        "level1": olci.get_sample("level1_fr")/olci.get_sample("level1_fr").name,
        "roi": {"x": slice(1717, 2151), "y": slice(2330, 2607)},
        "px": {"x": 100, "y": 100},
    }


@pytest.mark.parametrize('uncertainties', [False, True])
@pytest.mark.parametrize('roi_desc,roi', [
    ('roi1_small', {
        'scol':1930+120,
        'ecol':2100,
        'sline':2556+40,
        'eline':2650,
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
        Level1_OLCI(olci_level1, **roi),
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
            'bitmask',
            ]

    roi = {
        'scol':980,
        'ecol':980+1,
        'sline':2524,
        'eline':2524+1,
    }

    wav = [400,412,443,490,510,560,620,665,674,681,709,754,779,865,1020]
    l2 = run_atm_corr(
        Level1_OLCI(olci_level1, **roi),
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
            'bitmask',
            ]:
        data = getattr(l2, var)
        print(var, data)

    plt.legend()
    plt.grid(True)
    conftest.savefig(request)


@pytest.mark.parametrize('roi_desc,roi', [
    ('roi1_small', {
        'scol':1930+120,
        'ecol':2100,
        'sline':2556+40,
        'eline':2650,
    }),
    ('inland_water', {
        'scol':980,
        'ecol':1168,
        'sline':2524,
        'eline':2639,
    }),
])
def test_olci_write(roi_desc, roi):
    with TemporaryDirectory() as tmpdir:
        run_atm_corr(
            Level1_OLCI(olci_level1, **roi),
            Level2(outdir=tmpdir),
        )

def test_browse(request, testcase):
    """ 
    Test initialization, simple scene view
    """
    with timeit('polymer v5 init'):
        ds = run_polymer_dataset(
            autodetect.Level1(testcase['level1'])
        )
    plt.imshow(ds.Rtoa.sel(bands=865))
    plt.colorbar()
    conftest.savefig(request)


def test_v4(request, testcase):
    ds = run_v4(testcase)
    plot(request, testcase, ds)


@pytest.mark.parametrize("uncertainties", **pytest_utils.parametrize_dict({
    'unc': True,
    'nounc': False,
}))
def test_v5(request, uncertainties: bool, testcase: dict):
    ds = run_v5(testcase, uncertainties=uncertainties)
    plot(request, testcase, ds)


def test_v4_v5(testcase, request):
    """
    Test non-regression from v4 to v5
    """

    v4 = run_v4(testcase)
    v5 = run_v5(testcase)

    for b in v5.bands.values:
        diff(v4.rho_w.sel(bands=b), f'rho_w({b}) v4',
                v5.rho_w.sel(bands=b), f'rho_w({b}) v5')
        conftest.savefig(request)
    
    for varname in ['logchl', 'logfb', 'latitude', 'longitude', 'Rgli', 'Rnir']:
        diff(v4[varname], f'{varname} v4',
                v5[varname], f'{varname} v5')
        conftest.savefig(request)

    diff(v4.flags, 'bitmask v4',
            v5.flags, 'flags v5', percentile=0)
    conftest.savefig(request)

    for _ in diff_flags(
        v4.flags,
        v5.flags,
        None, # v4.flags.description.split(","),
        v5.flags.flag_meanings.split(),
        plot=True
    ):
        conftest.savefig(request)