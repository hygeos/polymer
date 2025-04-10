from pathlib import Path
from typing import Dict, Literal, Optional, Union

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


default_output_datasets = [
    "latitude",
    "longitude",
    "rho_w",
    "logchl",
    "logfb",
    "Rgli",
    "Rnir",
    "flags",
]

additional_output_datasets = [
    "Rtoa",
    "rho_gc",
    "Rprime",
    "Ratm",
]


def run_polymer(
    level1: Union[Path, str, xr.Dataset],
    *,
    roi: Optional[Dict] = None,
    file_out: Optional[Path|str] = None,
    ext: str = ".polymer.nc",
    dir_out: Optional[Path|str] = None,
    scheduler: str = "sync",
    split_bands: bool = True,
    output_datasets: Optional[list] = None,
    if_exists: Literal["skip", "overwrite", "backup", "error"] = "error",
    verbose: bool=True,
    **kwargs,
) -> Path:
    """
    Polymer: main function at file level

    Arguments:
        level1 is either a Path or a xr.Dataset (read with the eoread library)
        roi: definition of the region of interest. A dictionary such as
             {'x': slice(xmin, xmax, xstep), # or [xmin, xmax, xstep]
              'y': slice(ymin, ymax, ystep)}
        file_out (Path, optional): path to the output file. If not provided, use the
            two next arguments to determine the output file.
        ext (str): output filename extension
        dir_out (Path, optional): path to the output directory
        scheduler (str):
            "sync" for single-threaded
            "threads" for parallel processing (multiple threads)
        split_bands (bool): whether to split the output spectral bands into individual
            variables. Example: rho_w -> [rho_w_412, rho_w_443, ...]
        output_datasets: list of datasets to write to the output product.
            In case of empty list, print all available datasets and exit.
        if_exists: how to deal with existing output file
            ["skip", "overwrite", "backup", "error"]

    Returns:
        The path to the output product.
    """
    if isinstance(level1, (Path, str)):
        ds = Level1(Path(level1))
        basename = Path(level1).name
    elif isinstance(level1, xr.Dataset):
        ds = level1
        basename : str = ds.attrs['product_name']
    else:
        raise TypeError('Error in level1 dtype')

    if file_out is None:
        # determine file_out from dir_out and ext
        assert dir_out is not None
        file_out = Path(dir_out) / (basename + ext)
    assert file_out is not None
    
    if (roi is not None):
        ds = ds.isel({k: (v if isinstance(v, slice)
                      else slice(*v))
                      for k, v in roi.items()})

    # Run polymer main function
    ds = run_polymer_dataset(ds, **kwargs)

    # bands selection
    if output_datasets is None:
        output_datasets = default_output_datasets
    elif output_datasets == []:
        # print all available datasets and exit
        with xr.set_options(display_max_rows=150):
            print(ds)
        raise ValueError('Please provide a non-empty list of output_datasets. '
                         'The list of available datasets has been printed.')
    ds = ds[output_datasets]

    if split_bands:
        ds = split(ds, 'bands')

    with dask_config.set(scheduler=scheduler):
        to_netcdf(ds, filename=Path(file_out), if_exists=if_exists, verbose=verbose)
    
    return Path(file_out)


def init(ds: xr.Dataset, srf: xr.Dataset, params):
    """
    Initialize dataset `ds` for use with Polymer
    (in place)
    """
    init_Rtoa(ds)
    init_geometry(ds, scat_angle=True)

    apply_ancillary(
        ds,
        params.ancillary,
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


def run_polymer_dataset(ds: xr.Dataset, **kwargs) -> xr.Dataset:
    """
    Polymer: main function at dataset level
    """
    if "srf_getter" in kwargs:
        srf = rename(get_SRF(ds, **kwargs), ds.bands.values)
    else:
        # empty dictionary when srfs are not provided
        srf = xr.Dataset()

    params = Params(getattr(ds, 'sensor', None), **kwargs)

    init(ds, srf, params)
    
    apply_calib(ds, 'Rtoa', params.calib)

    ds = ds.sel(bands=params.bands_read()).chunk(bands=-1)

    if 'gsw_agg' in kwargs:
        apply_landmask(ds, **kwargs)

    init_uncertainties(ds, params)

    Gaseous_correction(ds, srf, input_var="Rtoa", **dict(params.items())).apply(
        method="map_blocks"
    )

    Rayleigh_correction(
        ds,
        bitmask_invalid=params.BITMASK_INVALID,
    ).apply(method="map_blocks")

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

