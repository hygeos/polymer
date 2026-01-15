#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import pytest
import tempfile
from matplotlib import pyplot as plt
from polymer.level1_msi import Level1_MSI
from polymer.level2_nc import Level2_NETCDF
from polymer import level2
from polymer.main import run_atm_corr
from eoread import eo
import xarray as xr
from core import env
from . import conftest


# @pytest.fixture(params=[
#     # 'S2B_MSIL1C_20230425T100029_N0509_R122_T31PFN_20230425T123406',   # Ancillary data consistency (Kerstin, 20230509)
#     # 'S2A_MSIL1C_20180328T144731_N0206_R139_T20PPC_20180328T212227',
#     # 'S2A_MSIL1C_20220202T110251_N0400_R094_T31UDS_20220202T130715',
#     # 'S2C_MSIL1C_20241213T101451_N9905_R022_T33TUL_20241213T121010',
# ])
# def msi_product(request):
#     return download_S2_google(request.param, env.getdir("DIR_SAMPLES") / "MSI")

@pytest.fixture
def msi_product() -> Path:
    # TODO: download using eoread.msi.get_sample
    # S2A_MSIL1C_20220202T110251_N0510_R094_T31UDS_20240512T150505.SAFE
    return env.getdir('LEVEL1_SAMPLE_MSI')

def test_instantiate(msi_product):
    print(Level1_MSI(msi_product))
    

@pytest.mark.parametrize('uncertainties', [True, False])
def test_msi(request, msi_product, uncertainties):

    l1 = Level1_MSI(
            msi_product,
            ancillary='ECMWFT',
            sline=0, eline=99,
            scol=0, ecol=99, resolution='20')
    with tempfile.TemporaryDirectory() as tmpdir:
        ret = run_atm_corr(
            l1,
            Level2_NETCDF(outdir=tmpdir,
                          datasets=(level2.default_datasets+level2.uncertainty_datasets) if uncertainties else level2.default_datasets),
            uncertainties=uncertainties,
        )
        print('Created file:', ret)
        ds = xr.open_dataset(ret.filename)
        assert 'Rgli' in ds

        plt.figure()
        plt.imshow(ds.rho_w_unc490 if uncertainties else ds.Rw490)
        plt.colorbar()

        conftest.savefig(request)


def test_msi_spectrum(request, msi_product):
    l1 = Level1_MSI(
            msi_product,
            sline=500, eline=510,
            scol=1000, ecol=1010)
    with tempfile.TemporaryDirectory() as tmpdir:

        l2 = run_atm_corr(l1, Level2_NETCDF(outdir=tmpdir, datasets=['Rw', 'Ratm', 'Rprime']))

        l2 = xr.open_dataset(l2.filename)
        l2 = eo.merge(l2, dim='wav', pattern=r'Rprime(\d+)', varname='Rprime')
        l2 = eo.merge(l2, dim='wav', pattern=r'Ratm(\d+)', varname='Ratm')
        l2 = eo.merge(l2, dim='wav', pattern=r'Rw(\d+)', varname='Rw')
        l2 = l2.isel(height=0, width=0)
        l2.Rw.plot(label='Rw')
        l2.Ratm.plot(label='Ratm')
        l2.Rprime.plot(label='Rprime')
        plt.grid(True)
        plt.legend()
        conftest.savefig(request)



