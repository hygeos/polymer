from pathlib import Path
from typing import Dict, Optional, Union

import xarray as xr
from core.interpolate import Nearest, interp
from core.save import to_netcdf
from core.tools import split, xrcrop
from dask import config as dask_config
from eoread.autodetect import Level1
from eoread.eo import init_geometry, init_Rtoa, raiseflag
from eoread.gsw import GSW
from eotools.apply_ancillary import apply_ancillary
from eotools.cm.basic import Cloud_mask
from eotools.gaseous_correction import Gaseous_correction
from eotools.glint import apply_glitter
from eotools.rayleigh_legacy import Rayleigh_correction
from eotools.srf import get_SRF, integrate_srf, rename

from polymer.common import L2FLAGS
from polymer.polymer_main import PolymerSolver
from polymer.uncertainties import init_uncertainties
from polymer.water import ParkRuddick
from polymer.params import Params


def run_polymer(
    level1: Union[Path, xr.Dataset],
    *,
    roi: Optional[Dict] = None,
    ext: str = ".polymer.nc",
    dir_out: Optional[Path] = None,
    file_out: Optional[Path] = None,
    scheduler='sync',
    split_bands=True,
    **kwargs,
):
    """
    Polymer: main function at file level

    Input:
        `level1` is either a Path or a xr.Dataset
    
    Arguments:
        roi: dictionary such as
             {'x': slice(xmin, xmax, xstep), # or [xmin, xmax, xstep]
              'y': slice(ymin, ymax, ystep)}

    Output:
        Either provide `file_out`or `dir_out`
    """
    if file_out is None:
        # determine file_out from dir_out and ext
        assert dir_out is not None
        file_out = dir_out / (level1.name + ext)
    assert file_out is not None

    if file_out.exists():
        raise FileExistsError(f"{file_out} exists")

    if isinstance(level1, Path):
        ds = Level1(level1)
    elif isinstance(level1, xr.Dataset):
        ds = level1
    
    if (roi is not None):
        ds = ds.isel({k: (v if isinstance(v, slice)
                      else slice(*v))
                      for k, v in roi.items()})

    # Run polymer main function
    ds = run_polymer_dataset(ds, **kwargs)

    # Add Rtoa865
    # assert "Rnir" not in ds
    # ds['Rnir'] = ds.Rtoa.sel(bands=kwargs['band_nir'])

    # bands selection
    output_datasets = [
        "latitude",
        "longitude",
        # "Rprime",
        "rho_w",
        # # "Ratm",
        # "Rtoa",
        "Rgli",
        "Rnir",
        "flags",
    ]  # FIXME: should not be hardcoded
    ds = ds[output_datasets]

    if split_bands:
        ds = split(ds, 'bands')

    with dask_config.set(scheduler=scheduler):
        to_netcdf(ds, filename=file_out)


def init(ds: xr.Dataset, srf: xr.Dataset, params, ancillary=None):
    """
    Initialize dataset `ds` for use with Polymer
    (in place)
    """
    init_Rtoa(ds)
    init_geometry(ds, scat_angle=True)

    apply_ancillary(
        ds,
        ancillary,
        {
                'horizontal_wind': 'm/s',
                'sea_level_pressure': 'hectopascals',
                'total_column_ozone': 'Dobson',
        })
    if 'altitude' not in ds:   # FIXME:
        ds['altitude'] = xr.zeros_like(ds.latitude)

    # Central wavelength
    if 'wav' not in ds:
        ds['wav'] = xr.DataArray(
            list(integrate_srf(srf, lambda x: x).values()),
            dims=['bands'],
            ).astype('float32')
    if ds.wav.dtype == 'float64':
        ds['wav'] = ds.wav.astype('float32')
    if 'cwav' not in ds:
        ds['cwav'] = ds.wav
        assert len(ds.wav.dims) == 1


def run_polymer_dataset(ds: xr.Dataset, ancillary=None, **kwargs) -> xr.Dataset:
    """
    Polymer: main function at dataset level
    """
    if "srf_getter" in kwargs:
        srf = rename(get_SRF(ds, **kwargs), ds.bands.values)
    else:
        # empty dictionary when srfs are not provided
        srf = xr.Dataset()

    params = Params(ds.sensor, **kwargs)

    init(ds, srf, params, ancillary=ancillary)
    
    apply_calib(ds, 'Rtoa', params.calib)

    ds = ds.sel(bands=params.bands_read()).chunk(bands=-1)

    if 'gsw_agg' in kwargs:
        apply_landmask(ds, **kwargs)

    init_uncertainties(ds, params)

    Gaseous_correction(ds, srf, input_var="Rtoa", **dict(params.items())).apply(
        method="map_blocks"
    )

    Rayleigh_correction(ds).apply(method='map_blocks')

    ds['Rnir'] = ds['Rprime_noglint'].sel(bands=params.band_cloudmask)

    Cloud_mask(ds,
               cm_input_var="Rprime",
               cm_band_nir=params.band_cloudmask,
               cm_flag_value=L2FLAGS["CLOUD_BASE"],
               cm_flag_name="CLOUD_BASE",
               ).apply(method='map_blocks')

    apply_glitter(ds)

    watermodel = ParkRuddick(
                    params.dir_common,
                    bbopt=params.bbopt,
                    min_abs=params.min_abs,
                    absorption=params.absorption)

    PolymerSolver(watermodel, params).apply(ds)

    return ds


def apply_landmask(ds: xr.Dataset, gsw_agg: int, **kwargs):
    gsw = GSW(agg=gsw_agg)

    # Crop gsw to the current location for optimization
    gsw = xrcrop(
        gsw,
        latitude=ds.latitude,
        longitude=ds.longitude,
    )

    # Compute gsw for interp
    gsw = gsw.compute()

    # Apply interp for nearest neighbour in lat/lon
    landmask = (
        interp(
            gsw,
            latitude=Nearest(ds.latitude),
            longitude=Nearest(ds.longitude),
        )
        < 50
    )

    raiseflag(ds.flags, 'LAND', 1, landmask)


def apply_calib(ds: xr.Dataset, varname: str, calib: dict|None):
    """
    Apply calibration coefficients `calib` to variable `varname` (in place)
    """
    if calib is not None:
        coeff = xr.DataArray([calib[x] for x in ds.bands.data], dims=['bands'])
        ds[varname] = ds[varname] * coeff

