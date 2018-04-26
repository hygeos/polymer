#!/usr/bin/env python
# -*- coding: utf-8 -*-

from polymer.main import run_atm_corr, Level1, Level2
from polymer.level2_hdf import Level2_HDF
from polymer.level2_nc import Level2_NETCDF
from polymer.level1_ascii import Level1_ASCII
from polymer.level1_nasa import Level1_NASA
from polymer.level1_olci import Level1_OLCI
from polymer.level1_msi import Level1_MSI
from pylab import plot


def example_meris():
    """
    Process a MERIS file
    using the generic (autodetecting) Level1 class
    and the generic level2 class (hdf4 by default)

    input file can be obtained with:
    wget http://www.brockmann-consult.de/beam/tutorials/BeamJavaWS/data/MERIS-Test-Data.zip
    unzip MERIS-Test-Data.zip
    """
    run_atm_corr(Level1('MER_RR__1P_TEST.N1'),
                 Level2(filename='output.hdf'),  # output file name can be explicit, otherwise it is inferred from Level1 file name
                 multiprocessing=-1,   # activate multiprocessing
                 )

    # NOTES:
    # * netcdf4 output can be selected with
    #   Level2(filename='output.nc', fmt='netcdf4')
    # * instead of the generic Level1 and Level2 you can use directly
    #   the appropriate Level1 and Level2 classes (see other examples)

def example_msi():
    """
    Sentinel-2 MSI processing

    Level1 can be specified as as granule name, like:
    * S2A_OPER_PRD_MSIL1C_20160318T145513.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20160318T232756_A003854_T19LDC_N02.01/
        (as downloaded on https://scihub.copernicus.eu/)
    * L1C_T51RTQ_A010954_20170728T024856/
        (as downloaded with Sentinelhub: https://github.com/sentinel-hub/sentinelhub-py)
    """
    run_atm_corr(
            Level1_MSI('S2A_OPER_PRD_MSIL1C_20160318T145513.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20160318T232756_A003854_T19LDC_N02.01/',
                       resolution='10'  # if not provided, use the 60m resolution
                       ),
            Level2_NETCDF()  # by default, output in the same folder as input
            )

def example_olci():
    """
    Sentinel-3 OLCI processing

    The name of the Level1 is the uncompressed product folder
    """
    run_atm_corr(
            Level1_OLCI('S3A_OL_1_ERR____20170114T124027_20170114T132421_20170115T181225_2633_013_152______LN1_O_NT_002.SEN3',
                        sline=9000, eline=10000),
            Level2_NETCDF(outdir='/tmp/')
            )


def example_modis():
    """
    MODIS processing
    including some custom parameters
    """
    run_atm_corr(Level1_NASA('A2004181120500.L1C', sensor='MODIS',
                              sline=1500, eline=2000, scol=100, ecol=500),
                 Level2_HDF(outdir='/data/',    # directory for result
                            ext='.polymer.hdf', # determine output filename from level1,
                                                # appending this extension
                            overwrite=True,     # overwrite existing output
                            compress=False,     # disactivate hdf compression
                            tmpdir='/tmp/',     # use temporary directory /tmp/
                            datasets=['Rw', 'latitude', 'longitude'],  # specify datasets to write
                           ),
                 # see params.py for exhaustive options
                 force_initialization=True,
                 normalize=0,
                 water_model='PR05',
                 )


def example_ascii():
    """
    Process an ASCII file (here for MERIS)
    using custom calibration coefficients
    returns in-memory level2 (do not write file)
    """
    l2 = run_atm_corr(Level1_ASCII('extraction.csv', square=5, sensor='MERIS'),
                      Level2('memory'),
                      force_initialization=True,
                      calib={
                         412: 1.01, 443: 0.99,
                         490: 1.0 , 510: 1.0,
                         560: 1.0 , 620: 1.0,
                         665: 1.0 , 681: 1.0,
                         709: 1.0 , 754: 1.0,
                         760: 1.0 , 779: 1.0,
                         865: 1.0 , 885: 1.0,
                         900: 1.0 ,
                         }  # or calib=None to set all coefficients to 1
                            # (default calibration: use per-sensor defaults as defined in
                            #  param.py)
                      )
    plot(l2.bands, l2.Rw[0,0,:])  # plot spectrum of pixel at (0,0)

def example_batch_processing():
    """
    Batch processing of Sentinel-3 OLCI
    """
    from glob import glob
    for filename in glob('path/to/level1/S3*.SEN3'):
        run_atm_corr(Level1_OLCI(filename),
                     Level2_NETCDF(outdir='/path/to/level2/'))


if __name__ == "__main__":
    example_msi()

