#!/usr/bin/env bash


conda env list

echo "This script will install the required dependencies in"
echo "the current anaconda environment."
echo "Please edit this script if you do not want to install all dependencies."
echo "If you want to create a new environment now, please"
echo "quit this script and use a command such as:"
echo "   conda create -n <envname>"
echo "for example:"
echo "   conda create -n polymer"
echo "and then activate this environment with:"
echo "   conda activate polymer"
echo


# mandatory dependencies
deps="python cython numpy pyhdf scipy netcdf4 pandas"

# MERIS support (optional)
deps="$deps avalentino::pyepr"

# S2-MSI support (optional)
deps="$deps glymur pyproj lxml"

# GSW support (global surface water landmask - optional)
deps="$deps gdal"

# ERA-Interim support (optional)
deps="$deps pygrib bioconda::ecmwfapi"

# python2 support (optional)
deps="$deps urllib3"


cmd="conda install -c conda-forge $deps"
echo 'The following command will be executed:'
echo "$cmd"
echo

read -r -p "Do you want to continue? [y/N] " response
if [[ "$response" =~ ^([yY])+$ ]]
then
    # proceed to the installation
    $cmd
fi
