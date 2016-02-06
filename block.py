#!/usr/bin/env python
# encoding: utf-8


class Block(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return 'block {}: size {}, offset {}'.format(self.id, self.size, self.offset)

    def set(self, name, value):
        self.__dict__.update({name: value})


