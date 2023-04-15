#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from tempfile import TemporaryDirectory

from matplotlib import pyplot as plt
import numpy as np
from polymer.gsw import GSW

from polymer.level2 import Level2
from . import conftest
import xarray as xr

import pytest
from polymer.ancillary_era5 import Ancillary_ERA5
from polymer.level1_prisma import Level1_PRISMA
from polymer.level2_nc import Level2_NETCDF
from polymer.main import run_atm_corr
from polymer import level2

# lidt of test cases
list_prisma_products = [
    {'filename': '/archive2/proj/ACIX3/202304/Garda/PRS_L1_STD_OFFL_20210721102700_20210721102705_0001.he5', # noqa
     'roi': {'sline': 150, 'eline': 550, 'scol': 150, 'ecol': 220},
     'pix': [200, 500],
     },
    {'filename': '/archive2/proj/ACIX3/202304/Garda/PRS_L1_STD_OFFL_20210917102722_20210917102726_0001.he5', # noqa
     'roi': {'sline': 350, 'eline': 550, 'scol': 0, 'ecol': 220},
     'pix': [200, 500],
     },
    {'filename': '/archive2/proj/ACIX3/202304/Geneve/PRS_L1_STD_OFFL_20200304104022_20200304104027_0001.he5', # noqa
     'roi': {'sline': 400, 'eline': 600, 'scol': 450, 'ecol': 650},
     'pix': [550, 500],
     },
    {'filename': '/archive2/proj/ACIX3/202304/Geneve/PRS_L1_STD_OFFL_20200530104010_20200530104014_0001.he5', # noqa
     'roi': {'sline': 400, 'eline': 600, 'scol': 450, 'ecol': 650},
     'pix': [550, 500],
     },
]

@pytest.mark.parametrize('level1', list_prisma_products)
def test_prisma_browse(level1, request):
    l1 = Path(level1['filename'])
    l2 = run_atm_corr(
        Level1_PRISMA(str(l1),
                      ancillary=Ancillary_ERA5(),
                      landmask=GSW(),
                      ),
        Level2('memory', datasets=['Rtoa']),
        partial=4,
    )
    
    plt.figure()
    plt.imshow(l2.Rtoa[:,:,l2.bands.index(859)],
               vmin=0,
               vmax=0.1)
    plt.colorbar()
    i0 = level1['pix'][0]
    j0 = level1['pix'][1]
    plt.plot([i0], [j0], 'r+')
    plt.title(l1.name)
    conftest.savefig(request)
    

@pytest.mark.parametrize('level1', list_prisma_products)
def test_prisma(level1, request):
    l1 = Path(level1['filename'])

    l2 = run_atm_corr(
        Level1_PRISMA(str(l1),
                      ancillary=Ancillary_ERA5(),
                      **level1['roi'],
                      landmask=GSW(),
                      ),
        Level2('memory',
               datasets=level2.default_datasets+level2.analysis_datasets
               ),
    )
    print(l2.shape)

    j0 = level1['pix'][0] - level1['roi']['scol']
    i0 = level1['pix'][1] - level1['roi']['sline']

    plt.figure()
    plt.imshow(l2.Rtoa[:,:,l2.bands.index(859)],
            vmin=0,
            vmax=0.1)
    plt.colorbar()
    plt.title('Rtoa865')
    plt.plot([i0], [j0], 'r+')
    conftest.savefig(request)

    plt.figure()
    plt.imshow(l2.Rw[:,:,l2.bands.index(559)],
            vmin=0,
            vmax=0.05)
    plt.colorbar()
    plt.title('rho_w(559)')
    conftest.savefig(request)

    plt.figure()
    plt.plot(
        l2.bands,
        l2.Rw[i0,j0,:],
        label='rho_w')
    plt.plot(
        l2.bands,
        l2.Rprime[i0,j0,:],
        label='rho_prime')
    plt.plot(
        l2.bands,
        l2.Rtoa[i0,j0,:],
        label='rho_toa')
    plt.grid(True)
    plt.title('spectrum visualization')
    conftest.savefig(request)
    
    

# TODO: check results identical with different offsets
# TODO: spectrum visualization

