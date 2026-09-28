#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Add CLOUD_EDGE to MODIS by post-processing.
"""

from pathlib import Path
from shutil import move
from sys import argv
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
from scipy import ndimage
from polymer.common import L2FLAGS, L2FLAGS_POLYMER
from polymer.utils import stdNxN


def modis_cloud_mask(
    polymer_file: str | Path,
    thres_Rcloud: float = 0.027,
    thres_Rcloud_std: float | None = 0.004,
    kernel: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    MODIS cloud mask post-processing from a Polymer L2 product

    Computes, from the variables stored in the Polymer product:

        cloud = (Rnir - Rgli > thres_Rcloud)
                | (Rnir_std3x3 > thres_Rcloud_std)   # if thres_Rcloud_std is not None

    where Rnir_std3x3 is the 3x3 local standard deviation of Rnir, computed over
    the pixels that are not L1_INVALID (saturated pixels are zero-filled and kept
    out of the windows).

    Note that the L1_INVALID pixels are included in the cloud mask (saturated
    pixels). No other L2 quality flag is used.

    - thres_Rcloud: Rnir-Rgli cloud threshold
    - thres_Rcloud_std: 3x3 std threshold on Rnir ; None disables it (default 0.004)
    - kernel: if not None, the cloud mask is dilated by a rectangular kernel of
      that size:

        cloud |= dilate(cloud)
    """

    ds = xr.open_dataset(str(polymer_file))
    try:
        Rnir = ds['Rnir'].values
        Rgli = ds['Rgli'].values
        bitmask = np.asarray(ds['bitmask'].values, dtype='int32')
    finally:
        ds.close()

    # L1_INVALID pixels are flagged as cloud sources as well (see docstring)
    l1_invalid = (bitmask & L2FLAGS['L1_INVALID']) != 0
    cloud = ((Rnir - Rgli) > thres_Rcloud) | l1_invalid

    if thres_Rcloud_std is not None:
        valid = ~l1_invalid
        Rnir_std3x3 = stdNxN(np.where(valid, Rnir, 0.0), 3, valid, fillv=0.)
        cloud |= Rnir_std3x3 > thres_Rcloud_std

    if kernel is not None:
        K = np.ones(kernel, dtype=bool)
        cloud |= ndimage.binary_dilation(cloud, structure=K)

    return cloud

def modis_cloud_mask_postprocess(filename: Path, dir_out: Path,
                                 thres_Rcloud: float = 0.027,
                                 thres_Rcloud_std: float | None = 0.004,
                                 kernel: tuple[int, int] | None = None,
                                 datasets: list[str] | None = None,
                                 compress: bool = True) -> None:
    """
    Add the MODIS cloud mask to `filename` and write it back to `dir_out`.

    The mask (see modis_cloud_mask) is stored as the CLOUD_EDGE bit
    (value 8, part of the standard BITMASK_REJECT test) of the product
    bitmask; the `bitmask.description` attribute is updated accordingly.

    - filename: input Polymer L2 product
    - dir_out: output directory
    - thres_Rcloud: Rnir-Rgli cloud threshold, passed to modis_cloud_mask
    - thres_Rcloud_std: 3x3 std threshold on Rnir, passed to modis_cloud_mask
      (None disables it)
    - kernel: straylight halo kernel size, passed to modis_cloud_mask
      (None: no halo)
    - datasets: list of output datasets. Default: all datasets.
    - compress: activate file compression
    """
    bit = L2FLAGS_POLYMER['CLOUD_EDGE']

    target = dir_out/filename.name
    if target.exists():
        raise IOError(f'Error: file {target} exists')

    print('Applying MODIS cloud mask:', filename)
    print('                       -->', target)

    cloud = modis_cloud_mask(
        filename,
        thres_Rcloud=thres_Rcloud,
        thres_Rcloud_std=thres_Rcloud_std,
        kernel=kernel,
    )

    ds = xr.open_dataset(filename)

    desc = ds.bitmask.description
    desc += f', CLOUD_EDGE:{bit}'
    ds.bitmask.attrs['description'] = desc
    ds['bitmask'].values += bit*cloud

    if datasets is not None:
        ds = ds[datasets]

    with TemporaryDirectory() as tmpdir:
        target_tmp = Path(tmpdir)/filename.name
        encoding = {var: dict(zlib=True, complevel=5)
                    for var in ds.data_vars} if compress else None
        ds.to_netcdf(target_tmp, encoding=encoding)
        ds.close()
        move(target_tmp, target)

if __name__ == '__main__':
    modis_cloud_mask_postprocess(Path(argv[1]), Path(argv[2]))
