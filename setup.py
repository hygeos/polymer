from distutils.core import setup
from Cython.Build import cythonize


debug=True


if debug:
    compiler_directives = {}
else:
    compiler_directives = {
            'boundscheck': False,
            'initializedcheck': False,
            'cdivision': True,
            }


setup(ext_modules = cythonize(['*.pyx'],
    compiler_directives=compiler_directives,
    annotate=debug,
    ))

