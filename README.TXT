
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

  SOFTWARE        REQUIRED FOR   URL
  python          all            http://www.python.org/
  cython          all            http://cython.org/

  PYTHON MODULES  REQUIRED FOR   URL
  numpy           all            https://pypi.python.org/pypi/numpy/
  scipy           all            https://pypi.python.org/pypi/scipy
  python-hdf4     all            https://pypi.python.org/pypi/python-hdf4
  pyepr           MERIS          https://pypi.python.org/pypi/pyepr/
  netCDF4         MODIS,SeaWiFS, https://pypi.python.org/pypi/netCDF4
                  VIIRS,OLCI,
                  level2_netcdf
  lxml            MSI            https://pypi.python.org/pypi/lxml
  glymur          MSI            https://pypi.python.org/pypi/Glymur/



1.2 Uncompressing
    -------------

Uncompress the "SRC" and "DATA" packages in POLYMER's root directory
$ tar xzvf polymer-src-v$VERSION.tar.gz
$ tar xzvf polymer-data-v$VERSION.tar.gz



1.3 COMPILATION
    -----------

The pyx files are cython files which need to be converted to C, then compiled.
A makefile is provided, so just type:
$ make



2) USAGE
   =====

2.1 How to run the algorithm
    ------------------------

There is a minimalistic command line interface:
$ ./polymer.py <level1> <level2>

Where <level1> is a level1 file or directory for any of the supported sensors,
and <level2> is the result to be generated.


More options are available by running polymer directly from your own python script.

    from polymer import polymer, Level1, Level2
    polymer(Level1('MER_RR__1PRACR20050501_092849_000026372036_00480_16566_0000.N1',
                   <other optional level1 arguments>),
            Level2('output.hdf',
                   <other optional level2 arguments>),
            <optional polymer arguments>)

See example.py for more details...


2.2 Output
    ------

The output files can be in format hdf4 or netcdf.
They contain the water reflectance (dimensionless, fully-normalized for sun and
sensor at nadir) and other self-explanatory parameters.

2.3 Flagging
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

