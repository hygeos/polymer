#!/usr/bin/env python

from core.download import download_url
from polymer.params import dir_static

"""
A script that fetches auxiliary data required bu Polymer
"""


def get_auxdata(URL = "https://download.hygeos.com/POLYMER/auxdata/"):
    for rpath in [
        # Common
        "generic/LUT.hdf",
        "common/no2_climatology.hdf",
        "common/trop_f_no2_200m.hdf",
        "common/morel_fq.dat",
        "common/AboveRrs_gCoef_w0.dat",
        "common/AboveRrs_gCoef_w10.dat",
        "common/AboveRrs_gCoef_w5.dat",
        "common/aph_bricaud_1995.txt",
        "common/aph_bricaud_1998.txt",
        "common/morel_buiteveld_bsw.txt",
        "common/palmer74.dat",
        "common/pope97.dat",
        "common/raman_westberry13.txt",
        "common/astarmin_average_2015_SLSTR.txt",
        "common/astarmin_average.txt",
        "common/Matsuoka11_aphy_Table1_JGR.csv",
        "common/k_oz.csv",
        "common/SOLAR_SPECTRUM_WMO_86",
        "common/vegetation.grass.avena.fatua.vswir.vh352.ucsb.asd.spectrum.txt",

        # MERIS
        "meris/smile/v2/sun_spectral_flux_rr.txt",
        "meris/smile/v2/central_wavelen_rr.txt",
        "meris/smile/v2/sun_spectral_flux_fr.txt",
        "meris/smile/v2/central_wavelen_fr.txt",

        # MODIS
        "modisa/HMODISA_RSRs.txt",

        # SeaWiFS
        "seawifs/SeaWiFS_RSRs.txt",

        # VIIRSN
        "viirs/VIIRSN_IDPSv3_RSRs.txt",

        # MSI
        "msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2A.csv",
        "msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2B.csv",
        "msi/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0_S2C.csv",

        # Landsat8 OLI
        "oli/Ball_BA_RSR.v1.2.xlsx",
    ]:
        download_url(URL + rpath, (dir_static / rpath).parent)


if __name__ == "__main__":
    get_auxdata()
