#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import sys

import numpy as np
import cv2

sys.path.insert(1,'../../LSHash/')

from lshash import LSHash

import leargist
from PIL import Image

COMPONENTS = 1000

DATA_PATH = "/mnt/hgfs/win/python"

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

def count_turn(arr):
    turn = 0
    base = arr[0]
    for i in arr[1:]:
        if i != base:
            turn = turn + 1
            base = i
    return turn

def count_stroke(arr):
    cnt = 0
    base = 0
    for i in arr:
        if i != base:
            if base == 1:
                cnt = cnt + 1
            base = i
    return cnt

def count_space(arr):
    cnt = 0
    base = 1
    for i in arr:
        if i != base:
            if base == 0:
                cnt = cnt + 1
            base = i
    return cnt

def extract_info(arr):
    total = float(len(arr))
    info = []
    cnt = 1
    base = arr[0]
    for i in arr[1:]:
        if i != base:
            info.append(round(cnt / total, 2) + base)
            cnt = 1
            base = i
        else:
            cnt = cnt + 1

    # append last one
    if cnt == total and base == 0:
        info.append(0)
    elif cnt > 1:
        info.append(round(cnt / total, 2) + base)

    # fill 0
    cnt = len(info)
    if cnt < 25:
        info = info + [0]*(25 - cnt)

    # add total info
    #info.append(total)

    return info

def extract_img(img):
    img = (img != 255) * 1

    for i in xrange(img.shape[0]):
        row = ""
        for j in xrange(img.shape[1]):
            row = row + ("#" if img[i,j] == 1 else "-")
        print row

    rows = np.vsplit(img, img.shape[0])
    cols = np.hsplit(img, img.shape[1])

    row_data = []
    for r in rows:
        row_data.append(extract_info(r.flatten()))

    col_data = []
    for c in cols:
        col_data.append(extract_info(c.flatten()))

    return (row_data, col_data)


def reconstruct_img(infos, size):
    rows = []
    for info in infos:
        if max(info) == 0:
            row = [0] * int(size)
        else:
            row = []
            for i in info:
                if i > 1:
                    row = row + [1] * (int(round((i - 1) * size)))
                else:
                    row = row + [0] * (int(round(i * size)))
        num = len(row)
        if num > size:
            row = row[:(size - num)]
        elif num < size:
            row = row + [0]*(size - num)
        rows.append(row)
    return rows

def cal_distance(arr):
    nonzero = np.nonzero(arr)[0]
    if nonzero.any():
        return nonzero[-1] - nonzero[0]
    return 0 

def load_img(fn):
    im_gray = cv2.imread(fn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im_gray = resize(im_gray, width=50, height=50)
    return im_gray

def get_img(fn):
    im_gray = cv2.imread(fn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    im_gray = resize(im_gray, width=50, height=50)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return features(im_bw) 

def features(img):

    img = (img != 255) * 1

    # only nonzero, will decrease the correct recognition ratio
    nonzero = np.nonzero(img)
    nonzero_img = img[min(nonzero[0]):max(nonzero[0])+1,min(nonzero[1]):max(nonzero[1])+1]

    rows = np.vsplit(img, img.shape[0])
    cols = np.hsplit(img, img.shape[1])

    h = nonzero_img.shape[0]
    w = nonzero_img.shape[1]

    horizontal_mean = np.mean(img, axis=0)
    vertical_mean = np.mean(img, axis=1)
    #print horizontal_min, horizontal_mean, horizontal_max, vertical_min, vertical_mean, vertical_max

    turn_each_row = [count_turn(i.flatten()) for i in rows]
    turn_each_col = [count_turn(i.flatten()) for i in cols]
    #print turn_each_row, turn_row_mean, turn_rows, turn_each_col, turn_col_mean, turn_cols

    stroke_each_row = [count_stroke(i.flatten()) for i in rows]
    stroke_each_col = [count_stroke(i.flatten()) for i in cols]
    #print stroke_each_row, stroke_each_col

    distance_each_row = [cal_distance(i.flatten()) for i in rows]
    distance_each_col = [cal_distance(i.flatten()) for i in cols]
    #print distance_each_row, distance_each_col

    turns_and_sum = [
        h/float(w), 
    ] + turn_each_row + turn_each_col + stroke_each_row + stroke_each_col + horizontal_mean.tolist() + vertical_mean.tolist()  + distance_each_row + distance_each_col # train rate: 100.000000  test rate: 94.708995
    
    return np.array(turns_and_sum) # [round(i, 1) for i in turns_and_sum] # 



BLOCKS = 3
ORIENTATIONS = (8, 8, 3)

DIMENSION = 401

#print leargist.color_gist(Image.open(DATA_PATH + "/trains//simhei.且.png"), nblocks=BLOCKS, orientations=ORIENTATIONS).shape
#print get_img(DATA_PATH + "/trains//simhei.且.png").shape
#exit()

################## train ##########################

wd = DATA_PATH + "/trains/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

lsh = LSHash(3, DIMENSION, num_hashtables=5)
print "extracting features ..."
t1 = time.time()
for i, fn in enumerate(files):
    #print unichr(ord(fn.split(".")[-2].decode("utf8")))
    #lsh.index(leargist.color_gist(Image.open(fn), nblocks=BLOCKS, orientations=ORIENTATIONS), extra_data=ord(fn.split(".")[-2].decode("utf8")))
    lsh.index(get_img(fn), extra_data=ord(fn.split(".")[-2].decode("utf8")))
t2 = time.time()
print "done. %d files took %0.3f ms" % (len(files), (t2 - t1) * 1000.0)

################## test ##########################

#rs = lsh.query(get_img(wd + '/simhei.且.png'), num_results=3, distance_func="euclidean")
#print [(unichr(r[0][1]), r[1]) for r in rs]
#rs = lsh.query(get_img(wd + '/simhei.且.png'), num_results=1, distance_func="euclidean")
#print unichr(rs[0][0][1])
#exit()

print "testing ..."


wd = DATA_PATH + "/trains/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

correct = 0
total = 0
t1 = time.time()
for i, fn in enumerate(files):
    total = total + 1
    rs = lsh.query(get_img(fn), num_results=1, distance_func="l1norm") 
    if rs and unichr(rs[0][0][1]) == fn.split(".")[-2].decode("utf8"):
        correct = correct + 1
    else:
        print unichr(rs[0][0][1]), " => ", fn.split(".")[-2].decode("utf8")
t2 = time.time()
print "train rate: %f, %d files took %0.3f ms" % (correct/float(total)*100, total, (t2 - t1) * 1000.0)



wd = DATA_PATH + "/tests/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

correct = 0
total = 0
t1 = time.time()
for i, fn in enumerate(files):
    total = total + 1
    #rs = lsh.query(get_img(fn), num_results=1, distance_func="cosine") # test rate: 91.326531, 196 files took 52901.431 ms
    #rs = lsh.query(get_img(fn), num_results=1, distance_func="l1norm") # test rate: 91.326531, 196 files took 35271.345 ms
    #rs = lsh.query(get_img(fn), num_results=1, distance_func="euclidean") # test rate: 90.816327, 196 files took 24904.888 ms
    #rs = lsh.query(get_img(fn), num_results=1, distance_func="true_euclidean") # test rate: 89.795918, 196 files took 17713.646 ms
    #rs = lsh.query(get_img(fn), num_results=1, distance_func="centred_euclidean") # test rate: 52.040816, 196 files took 9000.577 ms

    
    # BLOCKS = 1, ORIENTATIONS = (8, 8, 3), DIMENSION = 57, test rate: 89.285714, 196 files took 9997.003 ms
    # BLOCKS = 2, ORIENTATIONS = (8, 8, 3), DIMENSION = 228, test rate: 91.326531, 196 files took 17227.878 ms
    # BLOCKS = 3, ORIENTATIONS = (8, 8, 3), DIMENSION = 513, test rate: 98.469388, 196 files took 64944.190 ms
    # BLOCKS = 4, ORIENTATIONS = (8, 8, 4), DIMENSION = 960, test rate: 95.408163, 196 files took 47667.006 ms
    # BLOCKS = 5, ORIENTATIONS = (8, 8, 3), DIMENSION = 1425, test rate: 93.367347, 196 files took 71029.642 ms
    #rs = lsh.query(leargist.color_gist(Image.open(fn), nblocks=BLOCKS, orientations=ORIENTATIONS), num_results=1, distance_func="l1norm") 
    rs = lsh.query(get_img(fn), num_results=1, distance_func="l1norm") 
    if rs and unichr(rs[0][0][1]) == fn.split(".")[-2].decode("utf8"):
        correct = correct + 1
    else:
        print unichr(rs[0][0][1]), " => ", fn.split(".")[-2].decode("utf8")

t2 = time.time()
print "test rate: %f, %d files took %0.3f ms" % (correct/float(total)*100, total, (t2 - t1) * 1000.0)