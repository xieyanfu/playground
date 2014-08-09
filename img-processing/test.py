#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path
import sys  

import cv2
import util

# ===============================================================================

if __name__ == '__main__':
    
    if len(sys.argv) != 4:
        print "Usage:", sys.argv[0], " mode inputImg outputImg"
        exit()
        
    if sys.argv[1] not in ['moji', 'beitie']:
        print "Valid mode: moji and beitie"
        exit()

    if not os.path.isfile(sys.argv[2]):
        print sys.argv[2], " doesn't exist"
        exit()

    

    img = cv2.imread(sys.argv[2], cv2.CV_LOAD_IMAGE_GRAYSCALE)

    if sys.argv[1] == 'moji':

        #img = util.util('noise_removal', img)

        # Ä«¼£
        img = util.util('binarize', img)
        img = util.util('dilation', img)
        img = util.util('crop', img)
        img = util.util('resize', img)
        img = util.util('thin_zhangsuen', img)
        img = util.util('frame', img)

    else:
        # ±®Ìû
        img = util.util('sharpening', img)
        img = util.util('noise_removal', img)
        img = util.util('binarize', img)
        img = util.util('dilation', img)
        img = util.util('erosion', img)
        img = util.util('invert', img)
        img = util.util('border_removal', img)
        img = util.util('crop', img)
        img = util.util('resize', img)
        img = util.util('thin_zhangsuen', img)
        img = util.util('frame', img)

    cv2.imwrite(sys.argv[3], img)
    
    



