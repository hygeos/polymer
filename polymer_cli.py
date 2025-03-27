#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
A basic command line interface for Polymer
'''

from polymer import cli
import sys

if __name__ == "__main__":
    
    cli.main(sys.argv[1:])