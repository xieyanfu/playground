#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path
import sys   
import time

import cv2
import numpy as np
import zhangsuen

def print_timing(func):
    def wrapper(*arg, **kwargs):
        t1 = time.time()
        res = func(*arg, **kwargs)
        t2 = time.time()
        print '%s took %0.3f ms' % (arg[0], (t2 - t1) * 1000.0)
        return res
    return wrapper

@print_timing
def util(func, img, **kwargs):
    return eval(func)(img, **kwargs)

def binarize(img, **kwargs):
    avg = (int(np.amin(img)) + int(np.amax(img))) / 2
    (thresh, img) = cv2.threshold(img, avg, 255, cv2.THRESH_BINARY) # thresholding
    return img

def invert(img, **kwargs):
    return np.invert(img)

def crop(img, **kwargs):
    img = binarize(img)
    inverted = np.invert(img) # inverted so that black becomes white and white becomes black since we will check for nonzero values
    (i, j) = np.nonzero(inverted) # finding indexes of nonzero values
    if np.size(i) != 0: # in case the box contains no BLACK pixel(i.e. the box is empty such as checkbox)
        img = img[np.min(i):np.max(i),np.min(j):np.max(j)] # row column operation to extract the nonzero values
    return img

def resize(img, **kwargs):
    width = int(kwargs.get('width', 36))
    height = int(kwargs.get('height', 36))

    if img.shape[0]/img.shape[1] >= width/height:
        if img.shape[0] > width:
            dim = ((img.shape[1] * width) / img.shape[0], width)
        else:
            dim = (img.shape[0], img.shape[1])
    else:
        if (img.shape[1] > height):
            dim = (height, (img.shape[0] * height) / img.shape[1])
        else:
            dim = (img.shape[0], img.shape[1])

    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

def erosion(img, **kwargs):
    pixels = int(kwargs.get('kernel', 3))
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.erode(img, kernel, iterations = 1)

def dilation(img, **kwargs):
    pixels = int(kwargs.get('kernel', 3))
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.dilate(img,kernel,iterations = 1)

def opening(img, **kwargs):
    pixels = int(kwargs.get('kernel', 3))
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img, **kwargs):
    pixels = int(kwargs.get('kernel', 3))
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def gradient(img, **kwargs):
    pixels = int(kwargs.get('kernel', 3))
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def smoothing(img, **kwargs):
    kernel = int(kwargs.get('kernel', 5))
    return cv2.GaussianBlur(img, (kernel,kernel), 0)

def sharpening(img, **kwargs):
    smooth = cv2.GaussianBlur(img, (0,0), 3)
    return cv2.addWeighted(img, 1.5, smooth, -0.5, 0) 

def frame(img, **kwargs):
    width = int(kwargs.get('width', 40))
    height = int(kwargs.get('height', 40))

    top = abs(img.shape[0] - width) / 2
    bottom = height - img.shape[0] - top
    left = abs(img.shape[1] - height) / 2
    right = width - img.shape[1] - left

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)  

def border_removal(img, **kwargs):
    size = int(kwargs.get('size', 5))
    img[0:size,:] = 255   # first "top"  number of rows
    img[-size:,] = 255 # last "bottom" number of rows
    img[:,0:size] = 255  # first "left" number of columns
    img[:,-size:] = 255 # last "right" number of columns
    return img

def noise_removal(img, **kwargs):
    return cv2.bilateralFilter(img, 9, 75, 75)

def thin_zhangsuen(img, **kwargs):
    img = binarize(img)
    return zhangsuen.thin(img, False, False, False) 


# ===============================================================================

if __name__ == '__main__':

    operations = [
        'erosion', 'dilation', 'opening', 'closing', 'gradient', 'smoothing', 'sharpening', 'resize',
        'crop', 'binarize', 'invert', 'frame', 'noise_removal', 'border_removal', 'thin_zhangsuen',
    ]

    if len(sys.argv) < 4:
        print "Usage:", sys.argv[0], " operation inputImg outputImg [ param1=value1 param2=value2 ...]"
        print "     Valid operation: ", ', '.join(operations[:-1]), 'and ', operations[-1:][0]
        print
        print "     When operation is resize, you can specify width and height"
        print "     When operation is frame, you can specify width and height"
        print "     When operation is border_removal, you can specify size"
        print "     When operation is erosion, dilation, opening, closing, gradient or smoothing, you can specify kernel"
        exit()
        
    if sys.argv[1] not in operations:
        print "ValueError: operation is not valid"
        exit()

    if not os.path.isfile(sys.argv[2]):
        print sys.argv[2], " doesn't exist"
        exit()

    params = {}
    if len(sys.argv) > 4:
        args = sys.argv[4:]
        for arg in args:
            pairs = arg.split('=')
            if len(pairs) == 2:
                params[pairs[0]] = pairs[1]

    src = cv2.imread(sys.argv[2], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    dst = util(sys.argv[1], src, **params)
    cv2.imwrite(sys.argv[3], dst)












