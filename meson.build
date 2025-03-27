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

py = import('python').find_installation('python3')
cython = find_program('cython', required : true)
numpy_inc = run_command(py, ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
python_inc = run_command(py, ['-c', 'from sysconfig import get_paths as gp; print(gp()["include"])'], check: true).stdout().strip()

inc = [
    numpy_inc,
    python_inc,
    'polymer',
]

cython_args = [
    '-X', 'boundscheck=false',
    '-X', 'initializedcheck=false',
    '-X', 'cdivision=true',
    '-X', 'embedsignature=true',
    '-X', 'language_level=0',
]

libfresnel = static_library(
    'fresnel',
    'polymer/fresnel.c',  # Explicitly compile the C file
)

# CLUT
libclut = py.extension_module(
    'clut',
    'polymer/clut.pyx',
    include_directories: inc,
    c_args: [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-w',  # Suppress warnings (optional)
    ],
    cython_args: cython_args,
)

# NELDERMEAD
libneldermead = py.extension_module(
    'neldermead',
    'polymer/neldermead.pyx',
    include_directories: inc,
    c_args: [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-w',  # Suppress warnings (optional)
    ],
    cython_args: cython_args,
)

# POLYMER_MAIN
libpolymer_main = py.extension_module(
    'polymer_main',
    'polymer/polymer_main.pyx',
    include_directories: inc,
    c_args: [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-w',  # Suppress warnings (optional)
    ],
    cython_args: cython_args,
)

# WATER
libwater = py.extension_module(
    'water',
    'polymer/water.pyx',
    include_directories: inc,
    c_args: [
        '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
        '-w',  # Suppress warnings (optional)
    ],
    cython_args: cython_args,
    link_with: [libfresnel],  # Link the precompiled C code
)


# Custom target to copy .so to ./polymer
custom_target('move-clut-local',
    input: libclut,
    output: 'clut.so',
    command: [
        'cp',
        '@INPUT@',
        meson.current_source_dir() / 'polymer' / '@OUTPUT@'
    ],
    build_by_default: true
)

# Custom target to copy .so to ./polymer
custom_target('move-neldermead-local',
    input: libneldermead,
    output: 'neldermead.so',
    command: [
        'cp',
        '@INPUT@',
        meson.current_source_dir() / 'polymer' / '@OUTPUT@'
    ],
    build_by_default: true
)

# Custom target to copy .so to ./polymer
custom_target('move-polymer_main-local',
    input: libpolymer_main,
    output: 'polymer_main.so',
    command: [
        'cp',
        '@INPUT@',
        meson.current_source_dir() / 'polymer' / '@OUTPUT@'
    ],
    build_by_default: true
)

# Custom target to copy .so to ./polymer
custom_target('move-water-local',
    input: libwater,
    output: 'water.so',
    command: [
        'cp',
        '@INPUT@',
        meson.current_source_dir() / 'polymer' / '@OUTPUT@'
    ],
    build_by_default: true
)

# install package in site-packages
install_subdir(
    'polymer',                          # Source directory
    install_dir: py.get_install_dir(),  # Target: site-packages/
    strip_directory: false,             # Preserve `polymer/` as a subdir
    exclude_files: ['*.pyx', '*.c'],    # Exclude source files (keep .so/.py)
)