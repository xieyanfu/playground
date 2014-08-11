#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys

# realpath() will make your script run, even if you symlink it :)
ext = os.path.realpath(os.path.abspath('../font-to-img'))
if ext not in sys.path:
    sys.path.insert(0, ext)

from font_to_img import bitmap_font, vector_font
from chinese_chars import primary_chars

bitmap_font('../font-to-img/fonts/simsun.ttc', primary_chars, 'train/%s/simsun-12.png', img_size=14, font_size=12)

# train
#bitmap_font('../font-to-img/fonts/simsun.ttc', [unichr(c) for c in xrange(0x4E00, 0x9FA5)], 'train/%s/simsun-12.png', img_size=14, font_size=12)
#bitmap_font('../font-to-img/fonts/simsun.ttc', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simsun-17.png', img_size=34, font_size=17)
#vector_font('../font-to-img/fonts/simsun.ttc', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simsun-32.png', img_size=34, font_size=32)

#bitmap_font('../font-to-img/fonts/simhei.ttf', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simhei-12.png', img_size=34, font_size=12)
#bitmap_font('../font-to-img/fonts/simhei.ttf', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simhei-17.png', img_size=34, font_size=17)
#vector_font('../font-to-img/fonts/simhei.ttf', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simhei-32.png', img_size=34, font_size=32)

#bitmap_font('../font-to-img/fonts/simkai.ttf', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simkai-12.png', img_size=34, font_size=12)
#bitmap_font('../font-to-img/fonts/simkai.ttf', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simkai-17.png', img_size=34, font_size=17)
#vector_font('../font-to-img/fonts/simkai.ttf', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simkai-32.png', img_size=34, font_size=32)

#bitmap_font('../font-to-img/fonts/simyou.ttf', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simyou-12.png', img_size=34, font_size=12)
#bitmap_font('../font-to-img/fonts/simyou.ttf', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simyou-17.png', img_size=34, font_size=17)
#vector_font('../font-to-img/fonts/simyou.ttf', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'train/%s/simyou-32.png', img_size=34, font_size=32)

# test
#bitmap_font('../font-to-img/fonts/simsun.ttc', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'test/%s/simsun-12.png', img_size=34, font_size=12)
#bitmap_font('../font-to-img/fonts/simsun.ttc', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'test/%s/simsun-14.png', img_size=34, font_size=14)
#bitmap_font('../font-to-img/fonts/simsun.ttc', [unichr(c) for c in xrange(0x4E00, 0x51E8)], 'test/%s/simsun-32.png', img_size=34, font_size=32)
