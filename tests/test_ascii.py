#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from tests import conftest
from polymer.level1_ascii import Level1_ASCII
from polymer.main import run_atm_corr
from tempfile import TemporaryDirectory
from polymer.level2_nc import Level2_NETCDF
from polymer.level2 import default_datasets
import xarray as xr
from matplotlib import pyplot as plt

# file_ascii = '/home/francois/proj/SACSO/sacso/ftp.hygeos.com/Match-ups/Level-1/MDB_A_L1_AERONET_version2_20160401_20170630_Venise_MP15.csv'
file_ascii = 'tmp/venise_small.csv'

def test_ascii(request):
    """
    With additional datasets
    """
    with TemporaryDirectory() as tmpdir:
        wav2 = [412,443,490,510,530,550,555,560,665,668,865,1020]
        l1 = Level1_ASCII(
            file_ascii,
            sensor='OLCI',
            additional_headers=[f'rho_wn_IS_{x}' for x in wav2]
        )
        res = run_atm_corr(
            l1,
            Level2_NETCDF(
                outdir=tmpdir,
            )
        )

        ds = xr.open_dataset(res.filename)
        wav = [int(x.strip()) for x in ds.bands_rw[1:-1].split(',')]
        for i, irow in enumerate([10000, 20000, 30000, 40000]):
            plt.plot(wav, [ds.isel(height=irow, width=0)[f'Rw{x}'] for x in wav], ls='-', color=f'C{i}', label='Rw (Polymer)')
            plt.plot(wav2, [l1.csv[f'rho_wn_IS_{x}'][irow] for x in wav2], ls='--', color=f'C{i}', label='rho_wn_IS')
            plt.legend()
            plt.grid(True)
        conftest.savefig(request)
