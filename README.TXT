
  POLYMER
  =======

  ATMOSPHERIC CORRECTION OF SUN-GLINT
  CONTAMINATED OCEAN COLOUR OBSERVATIONS

  FRANÇOIS STEINMETZ
  PIERRE-YVES DESCHAMPS
  DIDIER RAMON
  (HYGEOS)


###########################################



This is the python/cython implementation of the Polymer atmospheric correction
algorithm.
http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-19-10-9783



1) INSTALLATION
   ============


1.1 DEPENDENCIES
    ------------

  SOFTWARE          REQUIRED FOR   URL
  python            all            http://www.python.org/
  cython            all            http://cython.org/
  gcc               all            https://gcc.gnu.org/

  PYTHON MODULES    REQUIRED FOR   URL
  numpy             all            https://pypi.python.org/pypi/numpy/
  scipy             all            https://pypi.python.org/pypi/scipy
  python-hdf4       all            https://pypi.python.org/pypi/python-hdf4
   (or pyhdf)                      https://pypi.python.org/pypi/pyhdf
  pandas            all            https://pypi.python.org/pypi/pandas
  pyepr             MERIS          https://pypi.python.org/pypi/pyepr/
  netCDF4           MODIS,SeaWiFS, https://pypi.python.org/pypi/netCDF4
                    VIIRS,OLCI,
                    level2_netcdf,
                    GSW
  lxml              MSI            https://pypi.python.org/pypi/lxml
  glymur            MSI            https://pypi.python.org/pypi/Glymur/
  pyproj            MSI            https://pypi.python.org/pypi/pyproj
  ecmwf-api-client  Ancillary_ERA  https://pypi.python.org/pypi/ecmwf-api-client
                    (on-the-fly download)
  pygrib            Ancillary_ERA  https://pypi.python.org/pypi/pygrib
  gdal              GSW            https://pypi.python.org/pypi/GDAL


NOTE: python 2.7 and python 3 (>=3.5) are both supported. Python 3 is recommended.

NOTE: The supported operating system is Linux. The code has not been tested under Windows or Mac OS X.


If you are using anaconda, see the script install-anaconda-deps.sh which help
installing all required dependencies in a dedicated environment (recommended method).


1.2 AUXILIARY DATA
    -------------

The auxiliary data can be downloaded through the following command:
$ make auxdata_all


1.3 COMPILATION
    -----------

The pyx files are cython files which need to be converted to C, then compiled.
A makefile is provided, so just type:
$ make

NOTE: the command 'make all' will download the auxiliary files and proceed to the compilation.


2) USAGE
   =====

2.1 How to run the algorithm
    ------------------------

There is a minimalistic command line interface polymer_cli.py
./polymer_cli.py <level1> <level2>

Where <level1> is a level1 file or directory for any of the supported sensors,
and <level2> is the result to be generated.

See ./polymer_cli.py -h for more help


More options are available by running polymer directly from your own python script.

    from polymer.main import run_atm_corr, Level1, Level2
    run_atm_corr(Level1('MER_RR__1PRACR20050501_092849_000026372036_00480_16566_0000.N1',
                        <other optional level1 arguments>),
                 Level2('output.hdf',
                        <other optional level2 arguments>),
                 <optional polymer arguments>)

See example.py for more details...


2.2 Ancillary data
    --------------

2.2.1 NASA Ancillary data
      -------------------

Ancillary data (ozone total column, wind speed, surface pressure) can be
provided to the level1 class through the class Ancillary_NASA (NASA files in
hdf4 format):

    from polymer.ancillary import Ancillary_NASA
    Level1(<filename>, ancillary=Ancillary_NASA())

NOTE: the class Ancillary_NASA has default options to automatically download and select
the closest available dataset, in the folder ANCILLARY/METEO/
This folder is initialized with the command 'make ancillary' or 'make all'.

For more information about the optional parameters, please look at the help of
Ancillary_NASA.

2.2.2 ERA Interim ancillary data
      --------------------------

Optionnally, the ancillary data (ozone total column, wind speed, surface
pressure) can be provided through ECMWF's global reanalysis ERA-Interim.

See https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-interim

The following python modules are necessary to interface Polymer with ERA-Interim:
    * ecmwf python api client (to download ERA-Interim files on the fly)
      A ECMWF key is necessary.
      See https://software.ecmwf.int/wiki/display/WEBAPI/Access+ECMWF+Public+Datasets
    * pygrib, to read the ERA-Interim files in grib format.

The ERA-Interim ancillary data is used by passing the class Ancillary_ERA to
the parameter ancillary of the Level1.

    from polymer.ancillary_era import Ancillary_ERA
    Level1(<filename>, ancillary=Ancillary_ERA())

By default, the closest data in time is automatically used, and downloaded on
the fly if necessary.
For more information, please look at the docstring of Ancillary_ERA.


2.3 Input data
    ----------

This section provides information about the supported file formats and sensors.

NOTE: The class Level1 (from polymer.level1 import Level1) autodetects the file
format and returns the appropriate specific level1 object (Level1_MERIS,
Level1_OLCI, etc).

2.3.1 MERIS/Envisat

    Both FF (reduced resolution) and FR (full resolution) are supported.

    Example:
    from polymer.level1_meris import Level1_MERIS
    Level1_MERIS('MER_RR__1PRACR20050501_092849_000026372036_00480_16566_0000.N1')
        # optional arguments: sline, eline, ancillary


2.3.2 OLCI/Sentinel3

    Both RR and FR are supported.
    The name of the Level1 product is the name of the directory.

    Example:
    from polymer.level1_olci import Level1_OLCI
    Level1_OLCI('S3A_OL_1_EFR____20170123T102747_20170123T103047_20170124T155459_0179_013_279_2160_LN1_O_NT_002.SEN3')
        # optional arguments: sline, eline, ancillary


2.3.3 MODIS/Aqua, SeaWiFS, VIIRS

    MODIS, SeaWiFS and VIIRS requires Level1C files as input.
    See next section about Level 1C files for more information.
    Example:
    from polymer.level1_nasa import *
    Level1_MODIS('A2010120124000.L1C')
    Level1_SeaWiFS('S2000116121145.L1C')
    Level1_VIIRS('V2013339115400.L1C')
        # optional arguments: sline, eline, scol, ecol, ancillary


2.3.4 MSI/Sentinel-2

    The name of the level1 product is the name of the granule directory.
    Warning: MSI data processing is in development.

    Example:
    from polymer.level1_msi import Level1_MSI
    Level1_MSI('S2A_OPER_PRD_MSIL1C_PDMC_20160504T225644_R094_V20160504T105917_20160504T105917.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20160504T163055_A004524_T30TXR_N02.02')
        # optional arguments: sline, eline, ancillary


2.3.5 Ascii input

    Polymer supports ascii (CSV) data input from multiple sensors through the level1 class Level1_ASCII.

2.3.6 Subsetted products produced by SNAP

    Level1_NETCDF can be used to read MERIS, OLCI or Sentinel2 products in
    netCDF4 format, written by SNAP, in particular when used for subsetting.


2.4 Level 1C files
    --------------

    NASA OBPG L1A and L1B don't include all necessary radiometric corrections.
    Thus it is necessary to apply l2gen with custom options to write the TOA
    reflectances into what we call "Level1C" product.

    The command line is typically:
    l2gen ifile=<level1a> ofile=<level1c> gain="1 1 1 1 1 1 1 1" oformat="netcdf4" l2prod="rhot_nnn polcor_nnn sena senz sola solz latitude longitude"

    See tools/make_L1C.py, which is a helper script to generate level 1c products.


2.5 Output
    ------

The output files can be in format hdf4 or netcdf.
They contain the water reflectance (dimensionless, fully-normalized for sun and
sensor at nadir) and other self-explanatory parameters.

2.6 Flagging
    --------

The recommended flagging of output pixels is the following:
Valid pixels should verify "bitmask & 1023 == 0"


3) Licencing information
   =====================

This software is available under the Polymer licence v1.1, available in the
LICENCE.TXT file.



4) Referencing
   ===========

When acknowledging the use of Polymer for scientific papers, reports etc please
cite the following reference:
François Steinmetz, Pierre-Yves Deschamps, and Didier Ramon, "Atmospheric
correction in presence of sun glint: application to MERIS", Opt. Express 19,
9783-9800 (2011), http://dx.doi.org/10.1364/OE.19.009783

