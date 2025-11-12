from matplotlib import pyplot as plt
import pytest
from polymer.main_v5 import run_polymer_dataset
from eoread.pace import get_sample, Level1B_PACE_OCI
from core.tests.conftest import savefig


@pytest.mark.parametrize("sample", [1, 2])
def test_pace_reader(request, sample):
    sample_level1 = get_sample(sample)

    l1 = Level1B_PACE_OCI(sample_level1["path"])

    l1.Rtoa.sel(bands=849).isel(**sample_level1["roi"]).plot()

    if "px" in sample_level1:
        px = sample_level1["px"]
        plt.plot([px["x"]], [px["y"]], "r+")

    savefig(request)


@pytest.mark.parametrize("sample", [1, 2])
def test_pace_polymer(request, sample):
    product_level1 = get_sample(sample)

    l1 = Level1B_PACE_OCI(product_level1["path"])

    l2 = run_polymer_dataset(l1).sel(product_level1["roi"])

    l2.rho_w.sel(bands=500, method="nearest").plot(vmin=0, vmax=0.05)

    savefig(request)


@pytest.mark.parametrize("sample", [1, 2])
def test_pace_polymer_singlepixel(request, sample):
    product_level1 = get_sample(sample)
    l1 = Level1B_PACE_OCI(product_level1["path"])

    if "px" not in product_level1:
        return

    y = product_level1["roi"]["y"].start + product_level1["px"]["y"]
    x = product_level1["roi"]["x"].start + product_level1["px"]["x"]
    l2 = run_polymer_dataset(
        l1.sel(
            y=slice(y, y + 1),
            x=slice(x, x + 1),
        ),
    ).isel(x=0, y=0)

    for var in [
        # "Rtoa",
        # "rho_gc",
        "Ratm",
        "Rprime",
        "rho_w",
    ]:
        l2[var].plot(label=var)  # type: ignore

    plt.plot(
        l2.bands_corr, [0 for _ in l2.bands_corr], "r.", label="bands used by Polymer AC"
    )

    plt.legend()
    plt.grid(True)
    savefig(request)
