from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

"""
Definition of top of atmosphere uncertainties
"""

def toa_uncertainties(block, params):
    mus = block.mus[...,None]

    if hasattr(block, 'F0'):
        F0 = block.F0
    else:
        # Interpolation of F0 in solar spectrum file
        solar_spectrum_file = Path(params.dir_common)/'SOLAR_SPECTRUM_WMO_86'
        solar_data = pd.read_csv(solar_spectrum_file, sep=' ')

        F0_interp = interp1d(solar_data['lambda(nm)'], solar_data['Sl(W.m-2.nm-1)'])
        F0 = F0_interp(block.cwavelen)*1000 # interpolate and convert to µm-1

    if hasattr(block, 'Ltoa'):
        Ltoa = block.Ltoa
    else:
        assert hasattr(block, 'Rtoa')
        Ltoa = (1/np.pi)*mus*F0*block.Rtoa

    Rtoa_var = (Ltoa/block.Ltyp) * (np.pi*block.sigma_typ/(F0*mus))**2
    block.Rtoa_var = Rtoa_var.astype('float32')
