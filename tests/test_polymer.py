

"""
New test file, with common testcases
"""

from pathlib import Path

import matplotlib.patches as patches
import pytest
from core.tests import pytest_utils
from core.tests.graphics import xrimshow

from tests.common import plot, plot_spectra, run_v4, run_v5
from tests.test_samples import sample
from eoread import autodetect
from polymer.ancillary import Ancillary_NASA
from polymer.ancillary_era5 import Ancillary_ERA5

from . import conftest

TESTCASES = {
    "OLCI": {
        "sample": "LEVEL1_SAMPLE_OLCI",
        "roi": {"x": slice(0, 500), "y": slice(3500, 4000)},
        "px": {"x": 200, "y": 3600},  # Absolute coordinates
    },
    "OLCI-inland_water": {
        "sample": "LEVEL1_SAMPLE_OLCI",
        "roi": {"x": slice(680, 1168), "y": slice(2424, 2739)},
        "px": {"x": 1000, "y": 2571},  # Absolute coordinates (center of ROI)
    },
    "MSI": {
        "sample": "LEVEL1_SAMPLE_MSI",
        "roi": {"x": slice(1000, 1400), "y": slice(500, 800)},
        "px": {"x": 1100, "y": 600},  # Absolute coordinates
    }
}

@pytest.fixture(params=TESTCASES.items(), ids=lambda x: x[0])
def testcase(request):
    name, tc = request.param
    level1 = Path(sample(tc["sample"]))
    # Handle nested SEN3/SAFE structure where data files are in a subdirectory
    # with the same name as the parent directory
    nested = level1 / level1.name
    if nested.is_dir():
        level1 = nested
    tc['level1'] = level1
    return tc


def test_browse(request, testcase):
    """
    simple scene view for all testcases
    """
    ds = autodetect.Level1(testcase['level1'], v1_compat=True)
    fig, ax, im = xrimshow(ds.Rtoa.sel(bands=865), vmin=0, vmax=0.2)
    fig.colorbar(im, ax=ax)
    rect = patches.Rectangle(
        (testcase["roi"]["x"].start, testcase["roi"]["y"].start),
        testcase["roi"]["x"].stop - testcase["roi"]["x"].start,
        testcase["roi"]["y"].stop - testcase["roi"]["y"].start,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    conftest.savefig(request)


@pytest.mark.parametrize("multiprocessing", **pytest_utils.parametrize_dict({
    'sync': 0,
    # 'para': -1,
}))
def test_v4(request, testcase, multiprocessing):
    """Run polymer v4 and plot results for the current testcase."""
    ds = run_v4(testcase, BITMASK_INVALID=0, multiprocessing=multiprocessing)   # FIXME: BITMASK_INVALID
    plot(request, testcase, ds)

@pytest.mark.parametrize("ancillary", **pytest_utils.parametrize_dict({
    'ERA5': Ancillary_ERA5,
    'NASA': Ancillary_NASA,
}))
def test_v4_ancillary(request, testcase, ancillary):
    """
    Run polymer v4 with external ancillary data (instead of the
    ancillary data embedded in the level1, when available)
    """
    ds = run_v4(testcase, level1_kwargs={'ancillary': ancillary()})
    plot(request, testcase, ds)


def test_v4_px(request, testcase: dict):
    """Run polymer v4 on a single central pixel only."""
    px = testcase["px"]
    roi = {
        "x": slice(px["x"], px["x"] + 1),
        "y": slice(px["y"], px["y"] + 1),
    }
    # Build a testcase with the 1x1 ROI so _px_roi computes (0, 0)
    tc = {**testcase, "roi": roi}
    ds = run_v4(testcase, roi=roi)
    plot_spectra(request, tc, ds)


@pytest.mark.parametrize("uncertainties", **pytest_utils.parametrize_dict({
    # 'unc': True,
    'nounc': False,
}))
@pytest.mark.parametrize("multiprocessing", **pytest_utils.parametrize_dict({
    'sync': 0,
    # 'para': -1,
}))
def test_v5(request, uncertainties: bool, testcase: dict, multiprocessing: int):
    ds = run_v5(testcase, uncertainties=uncertainties, multiprocessing=multiprocessing)
    plot(request, testcase, ds)


def test_v5_px(request, testcase: dict):
    """Run polymer v5 on a single central pixel only."""
    px = testcase["px"]
    roi = {
        "x": slice(px["x"], px["x"] + 1),
        "y": slice(px["y"], px["y"] + 1),
    }
    # Build a testcase with the 1x1 ROI so _px_roi computes (0, 0)
    tc = {**testcase, "roi": roi}
    ds = run_v5(testcase, roi=roi)
    plot_spectra(request, tc, ds)
