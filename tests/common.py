from pathlib import Path

import numpy as np
import xarray as xr
from eoread import eo
from matplotlib import pyplot as plt
from tempfile import TemporaryDirectory
from polymer.main_v5 import default_output_datasets, additional_output_datasets
from eoread.common import timeit
from polymer.main_v5 import run_polymer
from polymer.main import run_atm_corr
from polymer.level1 import Level1
from polymer.level2 import default_datasets, analysis_datasets
from polymer.level2_nc import Level2_NETCDF

from . import conftest

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

        ds = load_polymer(l2.filename)

    return ds


def run_v5(testcase, **kwargs) -> xr.Dataset:
    """
    Run polymer v5
    """
    with TemporaryDirectory() as tmpdir:
        with timeit('polymer v5'):
            file_v5 = run_polymer(
                testcase["level1"],
                dir_out=tmpdir,
                roi=testcase["roi"],
                split_bands=False,
                verbose=False,
                output_datasets=default_output_datasets+additional_output_datasets,
                **kwargs,
            )
        ds = load_polymer(file_v5)
    return ds

def load_polymer(filename: Path):
    """
    Load Polymer product as a xr.Dataset
    """
    ds = xr.open_dataset(filename, mask_and_scale={"bitmask": False})

    if "bitmask" in ds:
        # v4
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


def diff_flags(
    A, B, flagsA: list | None = None, flagsB: list | None = None, plot=False
):
    """
    Plot each individual flag in A and B, print their differences
    """
    if flagsA is not None:
        N = len(flagsA)
    elif flagsB is not None:
        N = len(flagsB)
    else:
        N = 8*A.dtype.itemsize
    for i in range(N):
        Ai = (A & 1<<i) >>i
        Bi = (B & 1<<i) >>i
        fA = f"{flagsA[i]}\t" if (flagsA is not None) else ''
        fB = f"{flagsB[i]}\t" if (flagsB is not None) else ''
        print(i, '\t', fA, fB,
                int((Ai & Bi).sum()), '\t',
                int((Ai & ~Bi).sum()), '\t',
                int((~Ai & Bi).sum()), '\t',
                int((~Ai & ~Bi).sum()),
                )
        if plot:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5*2, 4*1))

            ax = axes.flat[0]
            ax.imshow(Ai)
            if flagsA is not None:
                ax.set_title(flagsA[i])

            ax = axes.flat[1]
            ax.imshow(Bi)
            if flagsB is not None:
                ax.set_title(flagsB[i])

            yield


def diff(A, label_A: str, B, label_B: str, percentile=2, single_row=True):
    """
    Plot a diff of arrays A and B
    """
    vmin = min(np.nanpercentile(A, percentile), np.nanpercentile(B, percentile))
    vmax = max(np.nanpercentile(A, 100-percentile), np.nanpercentile(B, 100-percentile))

    if single_row:
        nrows, ncols = 1, 4
    else:
        nrows, ncols = 2, 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))

    # Plot each figure individually
    ax = axes.flat[0]
    im = ax.imshow(A, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(label_A)

    ax = axes.flat[1]
    im = ax.imshow(B, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(label_B)

    # Plot and histogram of the difference
    D = B - A
    vmin = np.nanpercentile(D, percentile)
    vmax = np.nanpercentile(D, 100-percentile)
    vmin, vmax = min(vmin, -vmax), max(vmax, -vmin)

    ax = axes.flat[2]
    im = ax.imshow(D, vmin=vmin, vmax=vmax, cmap='bwr')
    plt.colorbar(im, ax=ax)
    ax.set_title(f"'{label_B}' - '{label_A}'")

    ax = axes.flat[3]
    ax.hist(np.ravel(D), range=(vmin, vmax), bins=70)
    ax.set_title(f"'{label_B}' - '{label_A}'")

    plt.tight_layout()


def plot(request, testcase, ds: xr.Dataset):
    """
    Plot polymer result
    """
    for label, data, vmin, vmax in [
        ('rho_w(560)', ds.rho_w.sel(bands=560), 0, 0.07),
        ]:
        plt.figure()
        data.plot.imshow(vmin=vmin, vmax=vmax, origin='upper')
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