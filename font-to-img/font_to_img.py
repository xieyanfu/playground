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
        dest = img % char
        dir = os.path.dirname(dest)
        if not os.path.exists(dir):
            os.makedirs(dir)
        pic.save(dest)

@print_timing
def bitmap_font(font, chars, img, **kwargs):
    img_size = int(kwargs.get('img_size', 14))
    font_size = int(kwargs.get('font_size', 12))
    pygame.font.init()
    font = pygame.font.Font(font, font_size)
    linehight = font.get_linesize()
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
        dest = img % char
        dir = os.path.dirname(dest)
        if not os.path.exists(dir):
            os.makedirs(dir)
        pic.save(dest)


@print_timing
def fonts_on_one_img(fonts, chars, img, **kwargs):
    img_size = int(kwargs.get('img_size', 14))
    font_size = int(kwargs.get('font_size', 12))
    pygame.font.init()
    sio = StringIO.StringIO()
    pic = Image.new('RGB', (len(fonts) * img_size, len(chars) * img_size), (255, 255, 255))
    for fIdx, font in enumerate(fonts):
        font = pygame.font.Font(font, font_size)
        linehight = font.get_linesize()
        for cIdx, char in enumerate(chars):
            sio.truncate(0)
            (w, h) = font.size(char)
            if w == 0 or h == 0:
                continue
            text = font.render(char, True, (0, 0, 0), (255, 255, 255))
            pygame.image.save(text, sio)
            sio.seek(0)
            tmp = Image.open(sio)
            pic.paste(tmp, (img_size * fIdx + (img_size - w) / 2, img_size * cIdx + (img_size - h) / 2))
    dir = os.path.dirname(img)
    if not os.path.exists(dir):
        os.makedirs(dir)
    pic.save(img)


# ===============================================================================


if __name__ == '__main__':

    #bitmap_font('fonts/simsun.ttc', [unichr(c) for c in xrange(0x9F99, 0x9F9F)], 'img/%s/simsun-12.png', img_size=14, font_size=12)
    #bitmap_font('fonts/simsun.ttc', [unichr(c) for c in xrange(0x9F99, 0x9F9F)], 'img/%s/simsun-17.png', img_size=19, font_size=17)
    #vector_font('fonts/simsun.ttc', [unichr(c) for c in xrange(0x9F99, 0x9F9F)], 'img/%s/simsun-32.png', img_size=34, font_size=32)
    #fonts_on_one_img(['fonts/simsun.ttc', 'fonts/simyou.ttf'], [u'中', u'国'], 'img/test.png', img_size=20, font_size=17)
    fonts = ['fonts/书体坊颜体.ttf', 'fonts/华康龙门石碑W9.TTF', 'fonts/博洋柳体3500.TTF', 'fonts/康熙字典体.otf', 'fonts/方正姚体_GBK.ttf',
            'fonts/方正宋体S-超大字符集.TTF', 'fonts/方正楷体S-超大字符集.TTF', 'fonts/方正瘦金书_GBK.ttf', 'fonts/方正行楷_GBK.ttf',
            'fonts/方正魏碑_GBK.ttf', 'fonts/花園明朝體A.ttf', 
            'fonts/苹果丽黑(W6).otf', 'fonts/華康歐陽詢t.ttf']
    chars = [u'龹', u'龤', u'嚴', u'龙', u'龍', u'丁', u'与', u'舆', u'鳯', u'齾', u'麡', u'丑', u'丞', u'主', u'丹']
    fonts_on_one_img(fonts, chars, './test.png', img_size=50, font_size=46)

