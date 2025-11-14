project(
    'Polymer',
    'cython',
    'c',
    version: '0.1.0',
    default_options: [
        'buildtype=release',
        'warning_level=0',  # Completely disable warnings
        'werror=false',     # Don't treat warnings as errors
    ],
)

# Python and dependencies setup
py = import('python').find_installation('python3')
cython = find_program('cython', required : true)
numpy_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
python_inc = run_command(py, ['-c', 'from sysconfig import get_paths as gp; print(gp()["include"])'], check: true).stdout().strip()

# Common include directories
inc = [
    numpy_inc,
    python_inc,
    'polymer',
]

# Common Cython arguments
cython_args = [
    '-X', 'boundscheck=false',
    '-X', 'initializedcheck=false',
    '-X', 'cdivision=true',
    '-X', 'embedsignature=true',
    '-X', 'language_level=0',
]

# Common C arguments for extensions
common_c_args = [
    '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
    '-w',  # Suppress warnings (optional)
]

# Static library for Fresnel C code
libfresnel = static_library(
    'fresnel',
    'polymer/fresnel.c',
)

# Extension modules configuration
extension_modules = {
    'polymer_minimizer': {
        'source': 'polymer/polymer_minimizer.pyx',
        'link_with': [],
    },
    'clut': {
        'source': 'polymer/clut.pyx',
        'link_with': [],
    },
    'neldermead': {
        'source': 'polymer/neldermead.pyx',
        'link_with': [],
    },
    'polymer_main': {
        'source': 'polymer/polymer_main.pyx',
        'link_with': [],
    },
    'water': {
        'source': 'polymer/water.pyx',
        'link_with': [libfresnel],
    },
}

# Build extension modules
extension_targets = {}
foreach name, config : extension_modules
    extension_targets += {name: py.extension_module(
        name,
        config['source'],
        include_directories: inc,
        c_args: common_c_args,
        cython_args: cython_args,
        link_with: config['link_with'],
    )}
endforeach


# Create custom targets to copy .so (Linux/Mac) or .pyd (Windows) files to ./polymer directory
foreach name, target : extension_targets
    custom_target('move-@0@-local'.format(name.underscorify()),
        input: target,
        output: '@0@'.format(name),
        command: [
            py,
            '-c',
            'import shutil, sys, os; shutil.copyfile(sys.argv[1], os.path.join(sys.argv[2], os.path.basename(sys.argv[1])))',
            '@INPUT@',
            meson.current_source_dir() / 'polymer'
        ],
        build_by_default: true
    )
endforeach


# Install package in site-packages
install_subdir(
    'polymer',                          # Source directory
    install_dir: py.get_install_dir(),  # Target: site-packages/
    strip_directory: false,             # Preserve `polymer/` as a subdir
    exclude_files: ['*.pyx', '*.c'],    # Exclude source files (keep .so/.pyd/.py)
)