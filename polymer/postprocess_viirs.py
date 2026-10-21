#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Add CLOUD_EDGE to VIIRS by post-processing.
"""

from argparse import ArgumentParser
from pathlib import Path
from shutil import move
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr

from polymer.common import L2FLAGS, L2FLAGS_POLYMER
from polymer.utils import stdNxN


def viirs_cloud_mask(
    polymer_file: str | Path,
    thres_Rcloud: float | None = None,
    thres_Rnir: float = 0.05,
    thres_Rgli: float = 0.05,
    thres_Rcloud_std: float | None = 0.012,
) -> np.ndarray:
    """
    VIIRS cloud-edge mask post-processing from a Polymer L2 product

    Computes, from the variables stored in the Polymer product:

        cloud_edge = (Rnir > thres_Rnir)
                     | (Rgli > thres_Rgli)
                     | ((Rnir - Rgli) > thres_Rcloud)     # if not None
                     | (Rnir_std3x3 > thres_Rcloud_std)   # if not None

    where Rnir is the Rayleigh-corrected reflectance of the cloudmask band
    (862 nm for VIIRS, zero-filled on invalid/night pixels), Rgli is the
    modeled sun-glint reflectance (NaN where invalid), and Rnir_std3x3 is the
    3x3 local standard deviation of Rnir, computed over the pixels that are
    not L1_INVALID.

    - thres_Rcloud: (Rnir - Rgli) cloud threshold ; None disables it
      (default: disabled)
    - thres_Rnir: absolute cloud threshold on Rnir (default 0.05)
    - thres_Rgli: absolute cloud threshold on Rgli (default 0.05)
    - thres_Rcloud_std: 3x3 std threshold on Rnir ; None disables it
      (default 0.012)
    """

    ds = xr.open_dataset(str(polymer_file))
    try:
        Rnir = ds['Rnir'].values
        Rgli = ds['Rgli'].values
        bitmask = np.asarray(ds['bitmask'].values, dtype='int32')
    finally:
        ds.close()

    cloud_edge = ((Rnir > thres_Rnir)
                  | (Rgli > thres_Rgli))

    if thres_Rcloud is not None:
        cloud_edge |= (Rnir - Rgli) > thres_Rcloud

    if thres_Rcloud_std is not None:
        valid = (bitmask & L2FLAGS['L1_INVALID']) == 0
        Rnir_std3x3 = stdNxN(np.where(valid, Rnir, 0.0), 3, valid, fillv=0.)
        cloud_edge |= Rnir_std3x3 > thres_Rcloud_std

    return cloud_edge

def viirs_cloud_mask_postprocess(filename: Path, dir_out: Path,
                                 thres_Rcloud: float | None = None,
                                 thres_Rnir: float = 0.05,
                                 thres_Rgli: float = 0.05,
                                 thres_Rcloud_std: float | None = 0.012,
                                 datasets: list[str] | None = None,
                                 compress: bool = True) -> None:
    """
    Add the VIIRS cloud-edge mask to `filename` and write it back to `dir_out`.

    The mask (see viirs_cloud_mask) is stored as the CLOUD_EDGE bit
    (value 8, part of the standard BITMASK_REJECT test) of the product
    bitmask; the `bitmask.description` attribute is updated accordingly.

    - filename: input Polymer L2 product
    - dir_out: output directory (created if needed)
    - thres_Rcloud: (Rnir - Rgli) cloud threshold, passed to viirs_cloud_mask
      (None disables it ; default: disabled)
    - thres_Rnir: absolute threshold on Rnir, passed to viirs_cloud_mask
    - thres_Rgli: absolute threshold on Rgli, passed to viirs_cloud_mask
    - thres_Rcloud_std: 3x3 std threshold on Rnir, passed to viirs_cloud_mask
      (None disables it)
    - datasets: list of output datasets. Default: all datasets.
    - compress: activate file compression
    """
    bit = L2FLAGS_POLYMER['CLOUD_EDGE']

    dir_out.mkdir(parents=True, exist_ok=True)
    target = dir_out/filename.name
    if target.exists():
        raise IOError(f'Error: file {target} exists')

    print('Applying VIIRS cloud-edge mask:', filename)
    print('                             -->', target)

    cloud_edge = viirs_cloud_mask(
        filename,
        thres_Rcloud=thres_Rcloud,
        thres_Rnir=thres_Rnir,
        thres_Rgli=thres_Rgli,
        thres_Rcloud_std=thres_Rcloud_std,
    )

    ds = xr.open_dataset(filename)

    desc = ds.bitmask.attrs['description']
    if 'CLOUD_EDGE:' not in desc:
        desc += f', CLOUD_EDGE:{bit}'
        ds.bitmask.attrs['description'] = desc
    ds['bitmask'].values += bit*cloud_edge

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
    ap = ArgumentParser(
        description='Add the VIIRS cloud-edge mask (CLOUD_EDGE bit) to a '
                    'Polymer L2 product')
    ap.add_argument('filename', type=Path, help='input Polymer L2 product')
    ap.add_argument('dir_out', type=Path, help='output directory (created if needed)')
    ap.add_argument('--thres_Rcloud', type=float, default=None,
                    help='(Rnir - Rgli) cloud threshold ; omit to disable '
                         '(default: disabled)')
    ap.add_argument('--thres_Rnir', type=float, default=0.05,
                    help='absolute cloud threshold on Rnir (default: 0.05)')
    ap.add_argument('--thres_Rgli', type=float, default=0.05,
                    help='absolute cloud threshold on Rgli (default: 0.05)')
    ap.add_argument('--thres_Rcloud_std', default='0.012',
                    help='3x3 std threshold on Rnir ; "none" disables it '
                         '(default: 0.012)')
    ap.add_argument('--datasets', default=None,
                    help='comma-separated list of output datasets (default: all)')
    ap.add_argument('--no-compress', action='store_true',
                    help='disable netCDF compression (on by default)')
    args = ap.parse_args()

    thres_Rcloud_std = None if str(args.thres_Rcloud_std).lower() == 'none' \
        else float(args.thres_Rcloud_std)
    datasets = [d.strip() for d in args.datasets.split(',')] if args.datasets is not None else None

    viirs_cloud_mask_postprocess(
        args.filename,
        args.dir_out,
        thres_Rcloud=args.thres_Rcloud,
        thres_Rnir=args.thres_Rnir,
        thres_Rgli=args.thres_Rgli,
        thres_Rcloud_std=thres_Rcloud_std,
        datasets=datasets,
        compress=not args.no_compress,
    )
