Transitioning to Polymer v5
===========================

This repository contains a preview of Polymer v5, which includes a full rewrite of
the processing framework in Polymer: input, output and data structure management.

- The internal data structure is based on dask and xarray, which facilitates modularity,
  blockwise processing and parallel processing.
- The Polymer code base, besides this repository which contains the main Polymer atmospheric
  correction code, has been split into several modules:
  - The `eoread` package contains Level product readers
  - The `eotools` package contains some atmospheric correction modules, such as Rayleigh
    correction, correction for gaseous absorption, SRF management...
  - The `core` package contains various low level utilities

Please make sure that these packages are properly installed by following the installation section in the main README.md

The final v5 code will be released when all sensors are supported in v5.


How to run the code
-------------------

Both v4 and v5 main functions are present in the current codebase. In the future, after the final v5 release, the v4 codebase with be removed.

In v4:
```
from polymer.main import run_atm_corr
from polymer.level1 import Level1
from polymer.level2 import Level2

run_atm_corr(
    Level1('MER_RR__1P_TEST.N1'),
    Level2(filename='output.nc'),
    multiprocessing=-1,
)

```

In v5:
```
from polymer.main_v5 import run_polymer

run_polymer(
    'S3B_OL_1_[...].SEN3',
    dir_out = '/path/to/output_directory/',
)
```

Other arguments can be passed to the `run_polymer` function. Please refer to this function
docstring.

Supported sensors
-----------------

| Sensor          | v4  | v5  |
| --------------- | --- | --- |
| Sentinel-3 OLCI | X   | X   |
| Sentinel-2 MSI  | X   |     |
| MODIS AQUA      | X   |     |
| SeaWiFS         | X   |     |
| VIIRS           | X   |     |
| ENVISAT MERIS   | X   |     |
| PRISMA          | X   |     |
| Landsat8 OLI    | X   |     |
| ISS HICO        | X   |     |
| HYPSO-2 HSI     |     | X   |



Current limitations of v5
-------------------------

- Parallel processing (option scheduler='threads') is not yet supported in v5.