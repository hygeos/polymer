#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
from polymer.level2 import Level2_file
from pyhdf.SD import SD, SDC
import numpy as np
from os import remove
from os.path import exists, dirname, join, basename
import tempfile
from polymer.utils import safemove
from shutil import rmtree
from polymer.utils import pstr
from warnings import warn


class Level2_HDF(Level2_file):
    '''
    Level 2 in HDF4 format

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
    '''
    def __init__(self,
            filename=None, ext='.hdf', outdir=None,
            tmpdir=None, overwrite=False, datasets=None,
            compress=True):

        self.filename = filename
        self.overwrite = overwrite
        self.datasets = datasets
        self.compress = compress
        self.outdir = outdir
        self.ext = ext

        # temporary directories
        self.__tmpdir = tmpdir  # base dir
        self.tmpdir = None      # sub dir, should be removed
        self.tmpfiles = []      # temporary files

        self.sdslist = {}
        self.typeconv = {
                    np.dtype('float32'): SDC.FLOAT32,
                    np.dtype('float64'): SDC.FLOAT64,
                    np.dtype('int8'): SDC.INT8,
                    np.dtype('int16'): SDC.INT16,
                    np.dtype('int32'): SDC.INT32,
                    np.dtype('uint8'): SDC.UINT8,
                    np.dtype('uint16'): SDC.UINT16,
                    np.dtype('uint32'): SDC.UINT32,
                    }

    def init(self, level1):
        super(self.__class__, self).init(level1)

        if self.__tmpdir is None:
            tmpdir = dirname(self.filename)
        else:
            tmpdir = tempfile.mkdtemp(dir=self.__tmpdir, prefix='level2_hdf_tmp_')
            self.tmpdir = tmpdir

        self.tmpfilename = join(tmpdir, basename(self.filename) + '.tmp')

        if not self.compress:
            self.__hdf = SD(self.tmpfilename, SDC.WRITE | SDC.CREATE)
        else:
            # dict of temporary hdf objects
            self.__hdf = {}

    def hdf(self, name):
        '''
        returns a hdf4 object for a given dataset name
        '''
        if not self.compress:
            return self.__hdf
        else:
            if not name in self.__hdf:
                filename = '{}_{}.tmp'.format(self.tmpfilename, name)
                if exists(filename):
                    print('Removing file', filename)
                    remove(filename)
                self.__hdf[name] = SD(filename, SDC.WRITE | SDC.CREATE)
                self.tmpfiles.append(filename)

            return self.__hdf[name]


    def write_block(self, name, data, S, attrs):
        '''
        write data into sds name with slice S
        '''
        if data.ndim == 3:
            for i, b in enumerate(self.bands):
                sdsname = '{}{}'.format(name, b)
                self.write_block(sdsname, data[:,:,i], S, attrs)
            return

        # create dataset
        if name not in self.sdslist:
            dtype = self.typeconv[data.dtype]
            self.sdslist[name] = self.hdf(name).create(name, dtype, self.shape)
            if dtype in [SDC.FLOAT32, SDC.FLOAT64]:
                self.sdslist[name].setfillvalue(np.NaN)

            # set attributes
            for k, v in attrs.items():
                setattr(self.sdslist[name], k, v)

        # write
        self.sdslist[name][S] = data[:,:]


    def finish(self, params):
        if self.compress:
            hdf = SD(self.tmpfilename, SDC.WRITE | SDC.CREATE)

            for name in sorted(self.sdslist):
                if params.verbose:
                    print('Write compressed dataset {}'.format(name))
                sds = self.sdslist[name]

                dtype  = sds.info()[3]
                sds2 = hdf.create(name, dtype, self.shape)
                sds2.setcompress(SDC.COMP_DEFLATE, 9)
                sds2[:] = sds[:]

                sds2.endaccess()

        else:
            hdf = self.__hdf

        for name, sds in self.sdslist.items():
            sds.endaccess()

        # write attributes
        for k, v in params.items():
            setattr(hdf, str(k), pstr(v))

        if self.compress:
            # cleanup
            for name in self.__hdf:
                filename = '{}_{}.tmp'.format(self.tmpfilename, name)
                self.__hdf[name].end()
                remove(filename)

        hdf.end()

        # move to destination
        safemove(self.tmpfilename, self.filename)

    def attributes(self):
        attrs = {}
        attrs['l2_filename'] = self.filename
        return attrs

    def cleanup(self):
        if (self.__tmpdir is not None) and (self.tmpdir is not None):
            rmtree(self.tmpdir)
        for f in self.tmpfiles:
            if exists(f):
                remove(f)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


