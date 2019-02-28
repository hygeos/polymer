#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import datetime


def node(raw, data):
    if 'END_GROUP' in raw[0]:
        return raw[1:] 

    if 'GROUP' in raw[0]:
        key = raw[0].split('=')[1].strip()
        data[key] = {}
        raw = node(raw[1:], data[key])
        return raw

    else:
        key, value, raw = leaf(raw)
        data[key] = value
        raw = node(raw[1:], data)

    return raw

def leaf(raw):
    key = raw[0].split('=')[0].strip()
    value = raw[0].split('=')[1].strip()

    if value[0] == '"': # string
        value = value[1:-1]
    elif value[0] == '(':
        tmp = [float(a) for a in value[1:-1].split(',')]

        while value[-1] != ')': # list
            raw = raw[1:]
            value = raw[0].strip()
            tmp += [float(a) for a in value[1:-1].split(',')]

        value = tmp
    else:
        try:
            if '.' in value: #float
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            value = np.datetime64(value).astype(datetime.datetime)

    return key, value, raw

def parser(raw):
    data = {}
    subdata = data

    if 'GROUP' in raw[0] and 'GROUP' in raw[1]:
        key = raw[0].split('=')[1].strip()
        data[key] = {}
        subdata = data[key]
        raw = raw[1:]

    while len(raw)!=0:
        raw = node(raw, subdata)
        if raw[0][:3]=='END':
            break

    return data

def read_meta(filename):
    '''
    A parser for Landsat8 metadata and angles file in ODL (Object Desription Language)
    '''
    with open(filename) as pf:
        raw = pf.readlines()

    data = parser(raw)

    return data

