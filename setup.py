from distutils.core import setup
from distutils.extension import Extension
import numpy
from Cython.Build import cythonize


NAME = "Polymer"
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
        nthreads=4,
        ),
    include_dirs = [numpy.get_include()],
    )
