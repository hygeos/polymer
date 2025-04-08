"""
Tests of performance/non-regression in the development of Polymer v5
"""


from typing import Dict
from eoread import eo
from eoread.common import timeit
from eoread.eo import init_Rtoa
from matplotlib import pyplot as plt
import xarray as xr
from tempfile import TemporaryDirectory
from core import pytest_utils
import pytest
from pathlib import Path
from polymer.main import run_atm_corr
from polymer.level1 import Level1
from polymer.level2_nc import Level2_NETCDF
from polymer.level2 import default_datasets, analysis_datasets
from polymer.main_v5 import run_polymer, run_polymer_dataset
from eoread import msi, olci
from eoread import autodetect
from eoread.ancillary_nasa import Ancillary_NASA

from polymer.params import Params
from . import conftest


# TODO: better deal with that
config = {
    'OLCI': {
        # 'scheduler': 'threads',
    },
    'MSI': {
        'gsw_agg': 1,
        'ancillary': Ancillary_NASA(),
    }
}


testcases = {
    "OLCI": {
        "sensor": "OLCI",
        "level1": olci.get_sample("level1_fr")/olci.get_sample("level1_fr").name,
        "roi": {"x": slice(1500, 2000), "y": slice(500, 1000)},
    },
    "MSI": {
        "sensor": "MSI",
        "level1": msi.get_sample(),
        "roi": {"x": slice(0, 250), "y": slice(750, 1000)},
        "px": {"x": 100, "y": 100},
    },
}


def run_v4(testcase) -> xr.Dataset:
    assert testcase["level1"].exists()
    sx = testcase['roi']['x']
    sy = testcase['roi']['y']
    with TemporaryDirectory() as tmpdir:
        # Run polymer v4
        with timeit('polymer v4'):
            l2 = run_atm_corr(
                Level1(str(testcase["level1"]),
                    sline=sy.start,
                    eline=sy.stop,
                    scol=sx.start,
                    ecol=sx.stop,
                    ),
                Level2_NETCDF(
                    outdir=tmpdir,
                    datasets=default_datasets+analysis_datasets+['Rtoa_gc']
                    ),
            )

        ds = xr.open_dataset(l2.filename)

        # merge and rename variables
        ds = eo.merge(ds, 'bands', 'rho_w', r'Rw(\d+)')
        ds = eo.merge(ds, 'bands', 'Rtoa', r'Rtoa(\d+)')
        ds = eo.merge(ds, 'bands', 'rho_gc', r'Rtoa_gc(\d+)')
        ds = eo.merge(ds, 'bands', 'Rprime', r'Rprime(\d+)')
        ds = eo.merge(ds, 'bands', 'Ratm', r'Ratm(\d+)')
        ds = ds.rename({
            'bitmask': 'flags',
        }).rename_dims({
            'height': 'y',
            'width': 'x',
        })
    return ds

def run_v5(testcase) -> xr.Dataset:
    with TemporaryDirectory() as tmpdir:
        sensor = testcase["sensor"]
        # Run polymer v5
        file_v5 = Path(tmpdir) / "out_v5.nc"
        with timeit('polymer v5'):
            run_polymer(testcase["level1"],
                        file_out=file_v5,
                        roi=testcase['roi'],
                        split_bands=False,
                        **config[sensor])
        ds = xr.open_dataset(file_v5)
    return ds


def plot(request, testcase, ds: xr.Dataset):
    # Plot figures
    for label, data, vmin, vmax in [
        ('rho_w(560)', ds.rho_w.sel(bands=560), 0, 0.07),
        ]:
        plt.figure()
        data.plot.imshow(vmin=vmin, vmax=vmax)
        if 'px' in testcase:
            plt.plot([testcase['px']['x']], [testcase['px']['y']], 'r+')
        plt.title(label)
        conftest.savefig(request)

    # Plot spectra
    if 'px' in testcase:
        plt.figure()
        px = ds.isel(testcase['px'])
        for varname in [
            'Rtoa',
            'rho_gc',
            'Rprime',
            'Ratm',
            'rho_w',
            ]:
            if varname in px:
                px[varname].plot(label=varname)
        plt.axis(ymin=-0.02, ymax=0.18)
        plt.legend()
        plt.grid(True)
        conftest.savefig(request)


@pytest.mark.parametrize("testcase",
                         **pytest_utils.parametrize_dict(testcases))
def test_v5(request, testcase: Dict):
    ds = run_v5(testcase)
    plot(request, testcase, ds)


@pytest.mark.parametrize("uncertainties", **pytest_utils.parametrize_dict({
    'unc': True,
    'nounc': False,
}))
@pytest.mark.parametrize("compute", **pytest_utils.parametrize_dict({
    'compute': True,
    'nocompute': False,
}))
@pytest.mark.parametrize("testcase", **pytest_utils.parametrize_dict(testcases))
def test_v5_lazy(testcase: Dict, compute: bool, uncertainties: bool):
    """
    Check laziness: application of run_polymer_dataset should be fast
    """
    sensor = testcase["sensor"]
    l1 = autodetect.Level1(testcase["level1"])

    # Run polymer v5
    with timeit('polymer v5 init'):
        ds = run_polymer_dataset(
            l1,
            uncertainties=uncertainties,
            **config[sensor],
        )
    ds = ds.isel(**testcase['roi'])
    if compute:
        with timeit('polymer v5 compute'):
            ds.compute()


@pytest.mark.parametrize("testcase",
                           **pytest_utils.parametrize_dict(testcases))
def test_v4(request, testcase):
    ds = run_v4(testcase)
    plot(request, testcase, ds)


@pytest.mark.parametrize("testcase",
                         **pytest_utils.parametrize_dict(testcases))
def test_browse(request, testcase):
    l1 = autodetect.Level1(testcase["level1"])
    init_Rtoa(l1)
    l1 = l1.isel(testcase['roi'])
    l1.Rtoa.sel(bands=865).plot.imshow(origin='upper', vmin=0, vmax=0.2)
    conftest.savefig(request)
