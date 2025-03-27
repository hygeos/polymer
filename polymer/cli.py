import argparse
import sys
from polymer.main import run_atm_corr
from polymer.level1 import Level1
from polymer.level2 import Level2


def main(args=None):

    parser = argparse.ArgumentParser(
        description='''Polymer atmospheric correction, simple command line interface.
                       To pass additional parameters, it is advised to execute the
                       function run_atm_corr in a python script''')

    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('-fmt', choices=['hdf4', 'netcdf4', 'autodetect'],
                        default='autodetect',
                        help='Output file format')
    args = parser.parse_args()

    if args.fmt == 'autodetect':
        if args.output_file.endswith('.nc'):
            args.fmt = 'netcdf4'
        elif args.output_file.endswith('.hdf'):
            args.fmt = 'hdf4'
        else:
            print('Error, cannot detect file format from output file "{}"'.format(
                args.output_file))
            exit()

    run_atm_corr(Level1(args.input_file),
                 Level2(filename=args.output_file, fmt=args.fmt))