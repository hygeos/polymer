from distutils.core import setup
import numpy
from Cython.Build import cythonize


NAME = "Polymer"
DESC = "Polymer atmospheric correction algorithm (http://dx.doi.org/10.1364/OE.19.009783)"
SRC_DIR = 'polymer'
DEBUG=False
ANNOTATE=True


if DEBUG:
    compiler_directives = {
            'profile': True,
            'embedsignature': True,
            }
else:
    compiler_directives = {
            'boundscheck': False,
            'initializedcheck': False,
            'cdivision': True,
            'embedsignature': True,
            }

EXTENSIONS = cythonize([SRC_DIR + '/*.pyx'],
                       build_dir='build',
                       compiler_directives=compiler_directives,
                       annotate=ANNOTATE,
                       )

setup(
    name = NAME,
    description = DESC,
    ext_modules = EXTENSIONS,
    include_dirs = [numpy.get_include()],
    )

