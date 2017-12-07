#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from polymer.level2 import Level2_file
from polymer.utils import safemove
from netCDF4 import Dataset, default_fillvals
import tempfile
import numpy as np
from os.path import exists, dirname, join, basename
from os import remove
from shutil import rmtree


class Level2_NETCDF(Level2_file):
    '''
    Level2 in netCDF format

    filename: string
        if None, determine filename from level1 by using output directory
        outdir and extension ext
    outdir: output directory
    ext: output file extension
    overwrite: boolean
        overwrite existing file
    datasets: list or None
        list of datasets to include in level 2
        if None (default), use default_datasets defined in level2 module
    compress: activate compression
    tmpdir: path of temporary directory
    format: underlying file format as specified in netcdf's Dataset:
            one of 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC' or 'NETCDF3_64BIT'
    '''
    def __init__(self,
                 filename=None,
                 ext='.nc',
                 tmpdir=None,
                 outdir=None,
                 overwrite=False,
                 datasets=None,
                 compress=True,
                 format='NETCDF4_CLASSIC',
                 ):
        self.filename = filename
        self.overwrite = overwrite
        self.datasets = datasets
        self.compress = compress
        self.outdir = outdir
        self.initialized = False
        self.varlist = {}
        self.ext = ext
        self.__tmpdir = tmpdir   # base tmp dir
        self.tmpdir = None       # sub dir, should be removed
        self.tmpfilename = None
        self.format=format

    def init(self, level1):
        super(self.__class__, self).init(level1)

        if self.__tmpdir is None:
            tmpdir = dirname(self.filename)
        else:
            tmpdir = tempfile.mkdtemp(dir=self.__tmpdir, prefix='level2_netcdf4_tmp_')
            self.tmpdir = tmpdir

        self.tmpfilename = join(tmpdir, basename(self.filename) + '.tmp')

        self.root = Dataset(self.tmpfilename, 'w', format=self.format)


    def write_block(self, name, data, S, attrs={}):
        '''
        write data into sds name with slice S
        '''
        if data.ndim == 3:
            for i, b in enumerate(self.bands):
                sdsname = '{}{}'.format(name, b)
                self.write_block(sdsname, data[:,:,i], S, attrs)
            return

        if not self.initialized:

            self.root.createDimension('width', self.shape[1])
            self.root.createDimension('height', self.shape[0])
            self.initialized = True

        assert data.ndim == 2
        if self.format == 'NETCDF4_CLASSIC':
            # data type conversion
            if data.dtype == np.uint16:
                typ = np.int16
            elif data.dtype == np.uint32:
                typ = np.int32
            else:
                typ = data.dtype
        else:
            typ = data.dtype

        fill_value = default_fillvals[np.dtype(typ).kind+str(np.dtype(typ).itemsize)]

        if name not in self.varlist:
            # create variable
            self.varlist[name] = self.root.createVariable(
                    name, typ,
                    ['height', 'width'],
                    fill_value=fill_value,
                    zlib=self.compress)
            # set attributes
            self.varlist[name].setncatts(attrs)

        # deplace NaNs by default_fillvals
        data[np.isnan(data)] = fill_value

        # write block
        self.varlist[name][S[0], S[1]] = data


    def finish(self, params):
        # write attributes
        for k, v in params.items():
            self.root.setncatts({k: str(v)})
        self.root.close()

        # move to destination
        safemove(self.tmpfilename, self.filename)

    def attributes(self):
        attrs = {}
        attrs['l2_filename'] = self.filename
        attrs['l2_format'] = self.format
        return attrs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    def cleanup(self):
        if (self.tmpfilename is not None) and exists(self.tmpfilename):
            remove(self.tmpfilename)
        if self.tmpdir is not None:
            rmtree(self.tmpdir)
