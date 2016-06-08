from level2 import Level2_file
from pyhdf.SD import SD, SDC
import numpy as np


class Level2_HDF(Level2_file):
    '''
    Level 2 in HDF4 format

    filename: string or function
        function taking level1 filename as input, returning a string
    overwrite: boolean
        overwrite existing file
    datasets: list or None
        list of datasets to include in level 2
        if None (default), use Level2.default_datasets
    '''
    def __init__(self, filename, overwrite=False, datasets=None):

        self.filename = filename
        self.overwrite = overwrite
        self.datasets = datasets

        self.sdslist = {}
        self.typeconv = {
                    np.dtype('float32'): SDC.FLOAT32,
                    np.dtype('float64'): SDC.FLOAT64,
                    np.dtype('uint16'): SDC.UINT16,
                    np.dtype('uint32'): SDC.UINT32,
                    }

    def init(self, level1):
        super(self.__class__, self).init(level1)

        self.hdf = SD(self.filename, SDC.WRITE | SDC.CREATE)

    def write_block(self, name, data, S):
        '''
        write data into sds name with slice S
        '''

        # create dataset
        if name not in self.sdslist:
            dtype = self.typeconv[data.dtype]
            self.sdslist[name] = self.hdf.create(name, dtype, self.shape)

        # write
        self.sdslist[name][S] = data[:,:]


    def write(self, block):

        (yoff, xoff) = block.offset
        (hei, wid) = block.size
        S = (slice(yoff,yoff+hei), slice(xoff,xoff+wid))

        for d in self.datasets:

            # don't write dataset if not in block
            if d not in block.datasets():
                continue

            if block[d].ndim == 2:
                self.write_block(d, block[d], S)

            elif block[d].ndim == 3:
                for i, b in enumerate(block.bands):
                    sdsname = '{}{}'.format(d, b)
                    self.write_block(sdsname, block[d][:,:,i], S)
            else:
                raise Exception('Error ndim')

    def finish(self, params):
        for name, sds in self.sdslist.items():
            sds.endaccess()

        # write parameters
        # TODO

        self.hdf.end()



