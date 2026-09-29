#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Add CLOUD_EDGE to MODIS by post-processing.
"""

from argparse import ArgumentParser
from pathlib import Path
from shutil import move
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
    dilate_l1_invalid: tuple[int, int] | None = (3, 3),
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
    - dilate_l1_invalid: if not None, the L1_INVALID mask is first dilated by a
      rectangular kernel of that size. Default: (3, 3) ; None disables it
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
    if dilate_l1_invalid is not None:
        K = np.ones(dilate_l1_invalid, dtype=bool)
        l1_invalid |= ndimage.binary_dilation(l1_invalid, structure=K)
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
                                 dilate_l1_invalid: tuple[int, int] | None = (3, 3),
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
    - dilate_l1_invalid: L1_INVALID dilation kernel size, passed to
      modis_cloud_mask (default (3, 3); None: no dilation)
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
        dilate_l1_invalid=dilate_l1_invalid,
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
    def _pair(s: str) -> tuple[int, int] | None:
        if s.lower() == 'none':
            return None
        h, w = s.lower().split('x')
        return int(h), int(w)

    ap = ArgumentParser(
        description='Add the MODIS cloud mask (CLOUD_EDGE bit) to a Polymer L2 product')
    ap.add_argument('filename', type=Path, help='input Polymer L2 product')
    ap.add_argument('dir_out', type=Path, help='output directory (created if needed)')
    ap.add_argument('--thres_Rcloud', type=float, default=0.027,
                    help='Rnir-Rgli cloud threshold (default: 0.027)')
    ap.add_argument('--thres_Rcloud_std', default='0.004',
                    help='3x3 std threshold on Rnir ; "none" disables it (default: 0.004)')
    ap.add_argument('--kernel', default='none',
                    help='straylight halo kernel size HxW ; "none" for no halo (default: none)')
    ap.add_argument('--dilate_l1_invalid', default='3x3',
                    help='L1_INVALID dilation kernel size HxW ; "none" disables it (default: 3x3)')
    ap.add_argument('--datasets', default=None,
                    help='comma-separated list of output datasets (default: all)')
    ap.add_argument('--no-compress', action='store_true',
                    help='disable netCDF compression (on by default)')
    args = ap.parse_args()

    thres_Rcloud_std = None if str(args.thres_Rcloud_std).lower() == 'none' \
        else float(args.thres_Rcloud_std)
    datasets = [d.strip() for d in args.datasets.split(',')] if args.datasets is not None else None

    modis_cloud_mask_postprocess(
        args.filename,
        args.dir_out,
        thres_Rcloud=args.thres_Rcloud,
        thres_Rcloud_std=thres_Rcloud_std,
        kernel=_pair(args.kernel),
        dilate_l1_invalid=_pair(args.dilate_l1_invalid),
        datasets=datasets,
        compress=not args.no_compress,
    )
