
  POLYMER
  =======

  ATMOSPHERIC CORRECTION OF SUN-GLINT
  CONTAMINATED OCEAN COLOUR OBSERVATIONS

  François Steinmetz  
  Pierre-Yves Deschamps  
  Didier Ramon  
  [HYGEOS](www.hygeos.com)

--------------------------------------------------


This is the python/cython implementation of the Polymer atmospheric correction
algorithm.
http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-19-10-9783



## 1. Installation

### 1.1 Dependencies

Polymer is written in python3. It is highly recommended to use [anaconda](http://anaconda.org)
to install all required dependencies. The [miniconda](https://docs.conda.io/en/latest/miniconda.html) version is sufficient.
The file `environment.yml` can be used to install the dependencies, either in your current
anaconda environment, or in a new one.

To create a new anaconda environment (independent python installation) with Polymer dependencies:
```
  conda create -n polymer -c conda-forge mamba
  conda activate polymer
  mamba env update -f environment.yml
```

### 1.2 Auxiliary data

The auxiliary data can be downloaded with the following command:
```
$ make auxdata_all
```


### 1.3 Compilation

The pyx files are cython files which need to be converted to C, then compiled.
A makefile is provided, so just type:
```
$ make
```

NOTE: the command `make all` will download the auxiliary files and proceed to the compilation.


## 2. Usage

### 2.1 How to run the algorithm

There is a minimalistic command line interface `polymer_cli.py`
```
./polymer_cli.py <level1> <level2>
```

Where <level1> is a level1 file or directory for any of the supported sensors,
and <level2> is the result to be generated.

See `./polymer_cli.py -h` for more help


More options are available by running polymer directly from your own python script.

```python
from polymer.main import run_atm_corr, Level1, Level2
run_atm_corr(Level1('MER_RR__1PRACR20050501_092849_000026372036_00480_16566_0000.N1',
                    <other optional level1 arguments>),
                Level2('output.nc',
                    <other optional level2 arguments>),
                <optional polymer arguments>)
```

See `example.py` for more details...


**Multiprocessing**

One option is `multiprocessing`, which controls the number of threads available for Polymer processing (by default, multiprocessing is disactivated). To activate the processing on as many threads as there are cores on the CPU, pass the value `-1`:
```
run_atm_corr(..., multiprocessing=-1)
```

This option controls the parallelization of the core Polymer processing. However, Polymer relies on numpy, which can also use parallel processing, and results in a moderate usage of multiple cores. To also disactivate numpy multiprocessing, you can pass the environment variable `OMP_NUM_THREADS=1` (or use the [threadpoolctl](https://github.com/joblib/threadpoolctl) library)



### 2.2 Ancillary data

#### 2.2.1 NASA Ancillary data

Ancillary data (ozone total column, wind speed, surface pressure) can be
provided to the level1 class through the class Ancillary_NASA (NASA files in
hdf4 format):

```python
    from polymer.ancillary import Ancillary_NASA
    Level1(<filename>, ancillary=Ancillary_NASA())
```

**NOTE**: the class `Ancillary_NASA` has default options to automatically download and select
the closest available dataset, in the folder `ANCILLARY/METEO/`

This folder is initialized with the command `make ancillary` or `make all`.

For more information about the optional parameters, please look at the help of
`Ancillary_NASA`.


#### 2.2.2 ERA Interim ancillary data

Optionnally, the ancillary data (ozone total column, wind speed, surface
pressure) can be provided by ECMWF's global reanalysis ERA-Interim.

See https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era-interim

The following python modules are necessary to interface Polymer with ERA-Interim:
    * ecmwf python api client (to download ERA-Interim files on the fly)
      A ECMWF key is necessary.
      See https://software.ecmwf.int/wiki/display/WEBAPI/Access+ECMWF+Public+Datasets
    * pygrib, to read the ERA-Interim files in grib format.

The ERA-Interim ancillary data is used by passing the class Ancillary_ERA to
the parameter ancillary of the Level1.

```python
    from polymer.ancillary_era import Ancillary_ERA
    Level1(<filename>, ancillary=Ancillary_ERA())
```

By default, the closest data in time is automatically used, and downloaded on
the fly if necessary.
For more information, please look at the docstring of Ancillary_ERA.

#### 2.2.3 ERA5 ancillary data

The ancillary data can also be provided by ECMWF's ERA5 dataset:
https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5

The following python modules are required:
* cdsapi. A CDS API key is required.
  Please see https://cds.climate.copernicus.eu/api-how-to for more details.
* xarray

```python
    from polymer.ancillary_era5 import Ancillary_ERA5
    Level1(<filename>, ancillary=Ancillary_ERA5())
```

### 2.3 Input data

This section provides information about the supported file formats and sensors.

**NOTE**: The class `Level1` (`from polymer.level1 import Level1`) autodetects the file
format and returns the appropriate specific level1 object (Level1_MERIS, Level1_OLCI, etc).

#### 2.3.1 MERIS/Envisat

Both FF (reduced resolution) and FR (full resolution) are supported.

Example:
```python
from polymer.level1_meris import Level1_MERIS
Level1_MERIS('MER_RR__1PRACR20050501_092849_000026372036_00480_16566_0000.N1')
    # optional arguments: sline, eline, ancillary
```

#### 2.3.2 OLCI/Sentinel3

Both RR and FR are supported.
The name of the Level1 product is the name of the directory.

Example:
```python
from polymer.level1_olci import Level1_OLCI
Level1_OLCI('S3A_OL_1_EFR____20170123T102747_20170123T103047_20170124T155459_0179_013_279_2160_LN1_O_NT_002.SEN3')
    # optional arguments: sline, eline, scol, ecol, ancillary
```


#### 2.3.3 MODIS/Aqua, SeaWiFS, VIIRS

MODIS, SeaWiFS and VIIRS requires Level1C files as input.
See next section about Level 1C files for more information.

Example:
```python
from polymer.level1_nasa import *
Level1_MODIS('A2010120124000.L1C')
Level1_SeaWiFS('S2000116121145.L1C')
Level1_VIIRS('V2013339115400.L1C')
    # optional arguments: sline, eline, scol, ecol, ancillary
```


#### 2.3.4 MSI/Sentinel-2

The name of the level1 product refers to the path to the granule (in
the "GRANULE/" directory).

Example:

```python
from polymer.level1_msi import Level1_MSI
Level1_MSI('S2A_OPER_PRD_MSIL1C_PDMC_20160504T225644_R094_V20160504T105917_20160504T105917.SAFE/GRANULE/S2A_OPER_MSI_L1C_TL_SGS__20160504T163055_A004524_T30TXR_N02.02')
    # optional arguments: sline, eline, ancillary
```

#### 2.3.5 Ascii input

Polymer supports ascii (CSV) data input from multiple sensors through the level1 class Level1_ASCII.

#### 2.3.6 Subsetted products produced by SNAP

Level1_NETCDF can be used to read MERIS, OLCI or Sentinel2 products in
netCDF4 format, written by SNAP, in particular when used for subsetting.


### 2.4 Level 1C files

NASA OBPG L1A and L1B don't include all necessary radiometric corrections.
Thus it is necessary to apply `l2gen` with custom options to write the TOA
reflectances into what we call "Level1C" product.

The command line is typically:

```
l2gen ifile=<level1a> ofile=<level1c> gain="1 1 1 1 1 1 1 1" oformat="netcdf4" l2prod="rhot_nnn polcor_nnn sena senz sola solz latitude longitude"
```

See `tools/make_L1C.py`, which is a helper script to generate level 1c products.


### 2.5 Output

The output files can be in format hdf4 or netcdf.
They contain the water reflectance (dimensionless, fully-normalized for sun and
sensor at nadir) and other self-explanatory parameters.

### 2.6 Flagging

The Polymer flags are the following:

```
---------------------------------------------------------------------------------
| Flag name          | Flag value  | Description                                |
|--------------------|-------------|--------------------------------------------|
| LAND               | 1           | Land mask                                  |
| CLOUD_BASE         | 2           | Polymer's basic cloud mask                 |
| L1_INVALID         | 4           | Invalid level1 pixel                       |
| NEGATIVE_BB        | 8           | (deprecated flag)                          |
| OUT_OF_BOUNDS      | 16          | Retrieved marine parameters are outside    |
|                    |             | valid bounds                               |
| EXCEPTION          | 32          | A processing error was encountered         |
| THICK_AEROSOL      | 64          | Thick aerosol flag                         |
| HIGH_AIR_MASS      | 128         | Air mass exceeds 5                         |
| EXTERNAL_MASK      | 512         | Pixel was masked using external mask       |
| CASE2              | 1024        | Pixel was processed in "case2" mode        |
| INCONSISTENCY      | 2048        | Inconsistent result was detected           |
|                    |             | (atmospheric reflectance out of bounds     |
| ANOMALY_RWMOD_BLUE | 4096        | Excessive difference was found at 412nm    |
|                    |             | between Rw and Rwmod                       |
--------------------------------------------------------------------------------|
```

The recommended flagging of output pixels is the following ('&'
represents the bitwise AND operator):

```
------------------------------------------------------------------------------
| Sensor   | Recommended flagging         | Notes                            |
|          | (valid pixels)               |                                  |
|----------|------------------------------|----------------------------------|
| OLCI     | bitmask & 1023 == 0          |                                  |
|          |                              |                                  |
| MSI      | bitmask & 1023 == 0          |                                  |
|          |                              |                                  |
| MERIS    | bitmask & 1023 == 0          |                                  |
|          |                              |                                  |
| VIIRS    | bitmask & 1023 == 0          | Sun glint and bright (cloudy)    |
|          | and (Rnir<0.1)               | pixels are discarded             |
|          | and (Rgli<0.1)               |                                  |
|          |                              |                                  |
| SeaWiFS  | bitmask & 1023+2048 == 0     | The INCONSISTENCY flag           |
|          |                              | cleans up most noisy pixels      |
|          |                              |                                  |
| MODIS    | bitmask & 1023+4096 == 0     | The ANOMALY_RWMOD_BLUE removes   |
|          |                              | outliers appearing on MODIS      |
|          |                              | results at high SZA              |
-----------------------------------------------------------------------------|
```
Note: additional cloud masking using IdePix (https://github.com/bcdev/snap-idepix) is recommended.

## 3. Licencing information

This software is available under the Polymer licence v2.0, available in the
LICENCE.TXT file.



## 4. Referencing

When acknowledging the use of Polymer for scientific papers, reports etc please
cite the following reference:

* François Steinmetz, Pierre-Yves Deschamps, and Didier Ramon, "Atmospheric
  correction in presence of sun glint: application to MERIS", Opt. Express 19,
  9783-9800 (2011), http://dx.doi.org/10.1364/OE.19.009783

* François Steinmetz and Didier Ramon "Sentinel-2 MSI and Sentinel-3 OLCI consistent
  ocean colour products using POLYMER", Proc. SPIE 10778, Remote Sensing of the Open
  and Coastal Ocean and Inland Waters, 107780E (30 October 2018); https://doi.org/10.1117/12.2500232


