from distutils.core import setup
from Cython.Build import cythonize


debug=False
annotate=False


if debug:
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


setup(
    name = 'Polymer',
    ext_modules = cythonize(['polymer/*.pyx'],
            compiler_directives=compiler_directives,
            annotate=annotate,
        )
    )

