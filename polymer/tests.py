#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import
import numpy as np


def test_neldermead():
    ''' test module neldermead '''
    from polymer.neldermead import test
    test()

def test_water():
    ''' test module water '''
    from polymer.water import test
    test()

def test_clut():
    ''' test module clut '''
    from polymer.clut import test
    test()
