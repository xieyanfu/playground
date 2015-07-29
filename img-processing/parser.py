#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class Hcl:
    def get_char(self, hcl, position):
        f = open(hcl, 'rb')
        f.seek((position + 1) * 512)
        byte = f.read(512)
        bits = ['{0:08b}'.format(ord(m)) for m in byte]
        arr = np.zeros((64,64), dtype=np.uint8)
        for j in xrange(64):
            dat = bits[j*8:j*8 + 8]
            for k in xrange(8):
                for n in xrange(8):
                    arr[j, k*8+n] = dat[k][n]
        return arr
        
    def get_chars(self, hcl, positions):
        f = open(hcl, 'rb')
        chars = []
        for i in positions:
            f.seek((i + 1) * 512)
            byte = f.read(512)
            bits = ['{0:08b}'.format(ord(m)) for m in byte]
            arr = np.zeros((64,64), dtype=np.uint8)
            for j in xrange(64):
                dat = bits[j*8:j*8 + 8]
                for k in xrange(8):
                    for n in xrange(8):
                        arr[j, k*8+n] = dat[k][n]
            chars.append(arr)
        return chars

    def get_img(self, hcl, position, name, mode='1'):
        from PIL import Image
        arr = self.get_char(hcl, position)
        arr ^= 1
        image = Image.fromarray(arr * 255)
        image.convert(mode)
        image.save('%s.jpg' % name)



if __name__ == "__main__":

    parser = Hcl()
    parser.get_img('/mnt/hgfs/win/HCL2000/hh001.hcl', 2, 'test', mode='L')
