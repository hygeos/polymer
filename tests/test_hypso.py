from matplotlib import pyplot as plt
from core.files.fileutils import mdir
from core.env import getdir
from eoread.hypso import Level1_HYPSO
import pytest

from polymer.main_v5 import run_polymer, run_polymer_dataset
from tests.conftest import savefig


sample_product = getdir('DIR_SAMPLES')/'HYPSO'/'aeronetvenice_2025-03-04T10-38-05Z-l1c.nc'

def test_hypso(request):
    l1 = Level1_HYPSO(sample_product)

    l1_sub = l1.thin(x=4, y=4)
    l2 = run_polymer_dataset(l1_sub)
    
    px = l2.isel(x=50, y=100)

    plt.imshow(l2.Rtoa.sel(bands=791))
    plt.colorbar()
    savefig(request)

    # plt.imshow(l2.rho_w.sel(bands=589))
    # plt.colorbar()
    # savefig(request)

    for varname in ['Rtoa', 'Rprime', 'Ratm', 'rho_r', 'rho_w']:
        px[varname].plot(label=varname)
    plt.grid()
    plt.legend()
    savefig(request)


@pytest.mark.skip('Long')
def test_hypso_product():
    run_polymer(Level1_HYPSO(sample_product),
                dir_out=mdir('/tmp/hypso/'))
