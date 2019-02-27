import numpy as np
import datetime


class Landsat8_metadata:

    def __init__(self, filename):
        self._filename = filename

    def parser(self, raw):
        data = {}
        subdata = data
            
        if 'GROUP' in raw[0] and 'GROUP' in raw[1]:
            key = raw[0].split('=')[1].strip()
            data[key] = {}
            subdata = data[key]
            raw = raw[1:]

        while len(raw)!=0:
            raw = self.node(raw, subdata)
            if raw[0][:3]=='END':
                break

        return data

    def node(self, raw, data):

        if 'END_GROUP' in raw[0]:
            return raw[1:] 

        if 'GROUP' in raw[0]:
            key = raw[0].split('=')[1].strip()
            data[key] = {}
            raw = self.node(raw[1:], data[key])
            return raw

        else:
            key, value, raw = self.leaf(raw)
            data[key] = value
            raw = self.node(raw[1:], data)

        return raw

    def leaf(self, raw):
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

    def open(self):
        with open(self._filename) as pf:
            raw = pf.readlines()

        data = self.parser(raw)

        return data


if __name__=='__main__':
    filename = '/rfs/user/francois/TESTCASES/L8_OLI/Lake_Canada/LC80140282017275LGN00/LC08_L1TP_014028_20171002_20171014_01_T1_ANG.txt'
#    filename = '/rfs/user/francois/TESTCASES/L8_OLI/Lake_Canada/LC80140282017275LGN00/LC08_L1TP_014028_20171002_20171014_01_T1_MTL.txt'
    meta = Landsat8_metadata(filename)
    data = meta.open()
    print(data)
    print(data.keys())
