from distutils.core import setup
from distutils.extension import Extension
from pathlib import Path
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
            'language_level': '3',
            }
else:
    compiler_directives = {
            'boundscheck': False,
            'initializedcheck': False,
            'cdivision': True,
            'embedsignature': True,
            'language_level': '3',
            }


setup(
    name = NAME,
    description = DESC,
    ext_modules=cythonize(
        [Extension(
            '*',
            sources=[
                'polymer/*.pyx',
            ],
            define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
        )],
        build_dir='build',
        compiler_directives=compiler_directives,
        annotate=ANNOTATE,
        ),
    include_dirs = [numpy.get_include()],
    )
