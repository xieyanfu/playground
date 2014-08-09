#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path
import sys
import time

# for vector fonts
import Image, ImageFont, ImageDraw

# for bitmap fonts
import pygame
import StringIO

def print_timing(func):
    def wrapper(*arg, **kwargs):
        t1 = time.time()
        res = func(*arg, **kwargs)
        t2 = time.time()
        print '%s took %0.3f ms' % (func.__name__, (t2 - t1) * 1000.0)
        return res
    return wrapper

@print_timing
def vector_font(font, chars, img, **kwargs):
    img_size = int(kwargs.get('img_size', 20))
    font_size = int(kwargs.get('font_size', 18))
    font = ImageFont.truetype(font, font_size)
    for char in chars:
        (w, h) = font.getsize(char)
        offset = font.getoffset(char)
        pic = Image.new('RGB', (img_size,img_size), (255, 255, 255))
        draw = ImageDraw.Draw(pic)
        draw.text(((img_size - w - offset[0]) / 2, (img_size - h - offset[1]) / 2), char, font=font, fill=(0,0,0))
        pic.save(img % char)

@print_timing
def bitmap_font(font, chars, img, **kwargs):
    img_size = int(kwargs.get('img_size', 14))
    font_size = int(kwargs.get('font_size', 12))
    pygame.font.init()
    font = pygame.font.Font(font, font_size)
    sio = StringIO.StringIO()
    for char in chars:
        sio.truncate(0)
        (w, h) = font.size(char)
        text = font.render(char, True, (0, 0, 0), (255, 255, 255))
        pygame.image.save(text, sio)
        sio.seek(0)
        tmp = Image.open(sio)
        pic = Image.new('RGB', (img_size,img_size), (255, 255, 255))
        pic.paste(tmp, ((img_size - w) / 2, (img_size - h) / 2))
        pic.save(img % char)


# ===============================================================================


if __name__ == '__main__':

    bitmap_font('fonts/simsun.ttc', [unichr(c) for c in xrange(0x4E00, 0x4E05)], 'img/bitmap-%s.png')
    vector_font('fonts/simsun.ttc', [unichr(c) for c in xrange(0x4E00, 0x4E05)], 'img/vector-%s.png', img_size=20, font_size=18)

