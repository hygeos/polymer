from level2 import Level2_file
from netCDF4 import Dataset

class Level2_NETCDF(Level2_file):
    def __init__(self, filename, overwrite=False, datasets=None, compress=True):
        self.filename = filename
        self.overwrite = overwrite
        self.datasets = datasets
        self.compress = compress
        self.initialized = False
        self.varlist = {}

    def init(self, level1):
        super(self.__class__, self).init(level1)

        self.root = Dataset(self.filename, 'w', format='NETCDF4')

    def write_block(self, name, data, S):
        '''
        write data into sds name with slice S
        '''
        assert data.ndim == 2
        if name not in self.varlist:
            self.varlist[name] = self.root.createVariable(
                    name, data.dtype,
                    ['height', 'width'],
                    zlib=self.compress)

        self.varlist[name][S[0], S[1]] = data


    def write(self, block):
        (yoff, xoff) = block.offset
        (hei, wid) = block.size
        S = (slice(yoff,yoff+hei), slice(xoff,xoff+wid))

        if not self.initialized:

            self.root.createDimension('width', self.shape[1])
            self.root.createDimension('height', self.shape[0])
            self.initialized = True

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
        # TODO: write attributes
        self.root.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

