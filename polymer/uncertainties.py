from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import xarray as xr
from core.tools import Var

"""
Definition of top of atmosphere uncertainties
"""

vardef = Var('Rtoa_var', "float32", ('y', 'x', 'bands'))

def init_uncertainties(ds: xr.Dataset, params):
    """
    Initialize the input uncertainties
    """
    if params.uncertainties:
        ds['Ltyp'] = xr.DataArray(
                data=list(params.Ltyp.values()),
                dims=['bands'],
                coords={'bands': list(params.Ltyp)})
        ds['sigma_typ'] = xr.DataArray(
                data=list(params.sigma_typ.values()),
                dims=['bands'],
                coords={'bands': list(params.sigma_typ)})
        ds["Rtoa_var"] = xr.map_blocks(
            toa_uncertainties,
            ds[
                [
                    x
                    for x in ["F0", "Ltoa", "Rtoa", "mus", "sigma_typ", "Ltyp", "cwav"]
                    if x in ds
                ]
            ],
            template=vardef.to_dataarray(ds),
            kwargs={"dir_common": params.dir_common},
        )



def toa_uncertainties(block, dir_common):
    """
    Note: this function is either called with the v4 block structure, or with the
        v5 Dataset structure.
    """
    if not isinstance(block, xr.Dataset):
        mus = block.mus[...,None]
        cwav = block.cwavelen
    else:
        mus = block.mus
        cwav = block.cwav

    if hasattr(block, 'F0'):
        F0 = block.F0
    else:
        # Interpolation of F0 in solar spectrum file
        solar_spectrum_file = Path(dir_common)/'SOLAR_SPECTRUM_WMO_86'
        solar_data = pd.read_csv(solar_spectrum_file, sep=' ')

        F0_interp = interp1d(solar_data['lambda(nm)'], solar_data['Sl(W.m-2.nm-1)'])
        F0 = F0_interp(cwav)*1000 # interpolate and convert to µm-1
        if isinstance(block, xr.Dataset):
            F0 = xr.DataArray(F0, dims='bands')


    if hasattr(block, 'Ltoa'):
        Ltoa = block.Ltoa
    else:
        assert hasattr(block, 'Rtoa')
        Ltoa = (1/np.pi)*mus*F0*block.Rtoa

    Rtoa_var = (Ltoa/block.Ltyp) * (np.pi*block.sigma_typ/(F0*mus))**2

    if isinstance(block, xr.Dataset):
        return vardef.conform(Rtoa_var.astype('float32'), transpose=True)
    else:
        return Rtoa_var.astype('float32')
