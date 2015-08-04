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
    #return orientation(im_gray, 10, False).flatten().tolist() # train rate: 98.639456  test rate: 89.417989
    #return orientation(im_gray, 5, False).flatten().tolist() # train rate: 98.790627  test rate: 87.301587
    #return orientation(im_gray, 3, False).flatten().tolist() # train rate: 98.337113  test rate: 78.987150
    #angles = orientation(im_gray, 5, False)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #return features(im_bw) + angles.flatten().tolist() # angles = orientation(im_gray, 5, False) train rate: 100.000000  test rate: 93.424036
    #return features(im_bw) + np.mean(angles, axis=0).flatten().tolist() + np.mean(angles, axis=1).flatten().tolist() # angles = orientation(im_gray, 5, False) train rate: 100.000000  test rate: 93.801965
    return features(im_bw) # train rate: 100.000000  test rate: 93.348450

def features(img):

#    # settings for LBP
#    METHOD = 'uniform'
#    radius = 5
#    n_points = 8 * radius
#    #lbp = local_binary_pattern(img, n_points, radius, METHOD) # train rate: 72.108844  test rate: 30.914588
#    #lbp = ExtendedLBP(radius=9)(img) # train rate: 99.622071  test rate: 76.643991
#    #lbp = ExtendedLBP(radius=8)(img) # train rate: 99.773243  test rate: 78.155707
#    #lbp = ExtendedLBP(radius=7)(img) # train rate: 96.825397  test rate: 76.719577
#    #lbp = ExtendedLBP(radius=6)(img) # train rate: 100.000000  test rate: 83.673469
#    #lbp = ExtendedLBP(radius=5)(img) # train rate: 99.848828  test rate: 87.906274
#    #lbp = ExtendedLBP(radius=4)(img) # train rate: 99.546485  test rate: 87.981859
#    #lbp = ExtendedLBP(radius=3)(img) # train rate: 91.836735  test rate: 78.080121
#    #lbp = ExtendedLBP(radius=2)(img) # train rate: 92.970522  test rate: 75.510204
#    #lbp = ExtendedLBP(radius=1)(img) # train rate: 78.004535  test rate: 60.846561
#    #lbp = OriginalLBP()(img) # train rate: 84.278156  test rate: 63.794407
#    #lbp = LPQ()(img) # train rate: 100.000000  test rate: 83.975813
#    #lbp = LPQ(radius=1)(img) # train rate: 92.214664  test rate: 70.068027
#    #lbp = LPQ(radius=2)(img) # train rate: 99.546485  test rate: 79.213908
#    #lbp = LPQ(radius=3)(img) # train rate: 100.000000  test rate: 83.975813
#    #lbp = LPQ(radius=4)(img) # train rate: 100.000000  test rate: 88.662132
#    #lbp = LPQ(radius=5)(img) # train rate: 100.000000  test rate: 90.627362
#    lbp = LPQ(radius=6)(img) # train rate: 100.000000  test rate: 90.400605
#    #lbp = LPQ(radius=7)(img) # train rate: 100.000000  test rate: 90.627362
#    #lbp = LPQ(radius=8)(img) # train rate: 100.000000  test rate: 89.720333
#    #lbp = LPQ(radius=9)(img) # train rate: 100.000000  test rate: 87.150416
#    #lbp = LPQ(radius=10)(img) # train rate: 100.000000  test rate: 86.772487
#
#    n_bins = lbp.max() + 1
#    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
#    #print len(hist.flatten().tolist())
#    hist = hist.flatten().tolist()
#    fillup = 256
#    if len(hist) < fillup:
#        hist = hist + ([0] * (fillup - len(hist)))
    #return hist

    ###########################

    # ((img ^ 1) * 255)
    img = (img != 255) * 1
    #print img
    #print
    
    # only nonzero, will decrease the correct recognition ratio
    nonzero = np.nonzero(img)
    nonzero_img = img[min(nonzero[0]):max(nonzero[0])+1,min(nonzero[1]):max(nonzero[1])+1]

    rows = np.vsplit(img, img.shape[0])
    cols = np.hsplit(img, img.shape[1])
    #print img
    #print 

#    for i in xrange(img.shape[0]):
#        row = ""
#        for j in xrange(img.shape[1]):
#            row = row + ("#" if img[i,j] == 1 else "-")
#        print row

    h = nonzero_img.shape[0]
    w = nonzero_img.shape[1]
    total = img.shape[0]*img.shape[1]
    nonzero = np.count_nonzero(img)
    #print h, w, total, nonzero

    horizontal_sum = np.sum(img, axis=0)
    vertical_sum = np.sum(img, axis=1)
    #print horizontal_sum, vertical_sum

    horizontal_min = np.amin(img, axis=0)
    horizontal_mean = np.mean(img, axis=0)
    horizontal_max = np.amax(img, axis=0)
    vertical_min = np.amin(img, axis=1)
    vertical_mean = np.mean(img, axis=1)
    vertical_max = np.amax(img, axis=1)
    #print horizontal_min, horizontal_mean, horizontal_max, vertical_min, vertical_mean, vertical_max

    horizontal_min_sum = np.sum(horizontal_min)
    horizontal_mean_sum = np.sum(horizontal_mean)
    horizontal_max_sum = np.sum(horizontal_max)
    vertical_min_sum = np.sum(vertical_min)
    vertical_mean_sum = np.sum(vertical_mean)
    vertical_max_sum = np.sum(vertical_max)
    #print horizontal_min_sum, horizontal_mean_sum, horizontal_max_sum, vertical_min_sum, vertical_mean_sum, vertical_max_sum

    horizontal_min_mean = np.mean(horizontal_min)
    horizontal_mean_mean = np.mean(horizontal_mean)
    horizontal_max_mean = np.mean(horizontal_max)
    vertical_min_mean = np.mean(vertical_min)
    vertical_mean_mean = np.mean(vertical_mean)
    vertical_max_mean = np.mean(vertical_max)
    #print horizontal_min_mean, horizontal_mean_mean, horizontal_max_mean, vertical_min_mean, vertical_mean_mean, vertical_max_mean

    turn_horizontal_min = count_turn(horizontal_min)
    turn_horizontal_mean = count_turn(horizontal_mean)
    turn_horizontal_max = count_turn(horizontal_max)
    turn_vertical_min = count_turn(vertical_min)
    turn_vertical_mean = count_turn(vertical_mean)
    turn_vertical_max = count_turn(vertical_max)
    #print turn_horizontal_min, turn_horizontal_mean, turn_horizontal_max, turn_vertical_min, turn_vertical_mean, turn_vertical_max

    turn_each_row = [count_turn(i.flatten()) for i in rows]
    turn_row_mean = np.mean(turn_each_row)
    turn_rows = count_turn(turn_each_row)
    turn_each_col = [count_turn(i.flatten()) for i in cols]
    turn_col_mean = np.mean(turn_each_col)
    turn_cols = count_turn(turn_each_col)
    #print turn_each_row, turn_row_mean, turn_rows, turn_each_col, turn_col_mean, turn_cols
    #exit()
    #print turn_each_row
    #print turn_each_col

    stroke_each_row = [count_stroke(i.flatten()) for i in rows]
    stroke_each_col = [count_stroke(i.flatten()) for i in cols]
    #print stroke_each_row, stroke_each_col

    space_each_row = [count_space(i.flatten()) for i in rows]
    space_each_col = [count_space(i.flatten()) for i in cols]
    #print space_each_row, space_each_col

    distance_each_row = [cal_distance(i.flatten()) for i in rows]
    distance_each_col = [cal_distance(i.flatten()) for i in cols]
    #print distance_each_row, distance_each_col

    row_turn_0 = turn_each_row.count(0)
    row_turn_1 = turn_each_row.count(1)
    row_turn_3 = len([i for i in turn_each_row if i > 1 and i <= 3])
    row_turn_5 = len([i for i in turn_each_row if i > 3 and i <= 5])
    row_turn_7 = len([i for i in turn_each_row if i > 5 and i <= 7])
    row_turn_9 = len([i for i in turn_each_row if i > 7 and i <= 9])
    row_turn_11 = len([i for i in turn_each_row if i > 9 and i <= 11])
    row_turn_15 = len([i for i in turn_each_row if i > 11 and i <= 15])
    row_turn_21 = len([i for i in turn_each_row if i > 15 and i <= 21])
    row_turn_more = len([i for i in turn_each_row if i > 21])
    #print row_turn_0, row_turn_1, row_turn_3, row_turn_5, row_turn_7, row_turn_9, row_turn_11, row_turn_15, row_turn_21, row_turn_more

    col_turn_0 = turn_each_col.count(0)
    col_turn_1 = turn_each_col.count(1)
    col_turn_3 = len([i for i in turn_each_col if i > 1 and i <= 3])
    col_turn_5 = len([i for i in turn_each_col if i > 3 and i <= 5])
    col_turn_7 = len([i for i in turn_each_col if i > 5 and i <= 7])
    col_turn_9 = len([i for i in turn_each_col if i > 7 and i <= 9])
    col_turn_11 = len([i for i in turn_each_col if i > 9 and i <= 11])
    col_turn_15 = len([i for i in turn_each_col if i > 11 and i <= 15])
    col_turn_21 = len([i for i in turn_each_col if i > 15 and i <= 21])
    col_turn_more = len([i for i in turn_each_col if i > 21])
    #print col_turn_0, col_turn_1, col_turn_3, col_turn_5, col_turn_7, col_turn_9, col_turn_11, col_turn_15, col_turn_21, col_turn_more

    # fit
    deg = 15
    x = np.arange(1, 51, 1)
    stroke_each_row_fit = np.polyfit(x, np.array(stroke_each_row), deg)
    stroke_each_col_fit = np.polyfit(x, np.array(stroke_each_col), deg)
    turn_each_row_fit = np.polyfit(x, np.array(turn_each_row), deg)
    turn_each_col_fit = np.polyfit(x, np.array(turn_each_col), deg)
    horizontal_sum_fit = np.polyfit(x, np.array(horizontal_sum), deg)
    vertical_sum_fit = np.polyfit(x, np.array(vertical_sum), deg)
    distance_each_row_fit = np.polyfit(x, np.array(distance_each_row), deg)
    distance_each_col_fit = np.polyfit(x, np.array(distance_each_col), deg)

#    #x = np.arange(1, 17, 1)
#    #y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
#    #x = np.arange(1, 51, 1)
#    y = np.array(distance_each_col)
#    print x
#    print y
#    print 
#    
#    #第一个拟合，自由度为3
#    z1 = np.polyfit(x, y, 3)
#    # 生成多项式对象
#    p1 = np.poly1d(z1)
#    print(z1)
#    print(p1)
#    print 
#    
#    # 第二个拟合，自由度为6
#    z2 = np.polyfit(x, y, 15)
#    # 生成多项式对象
#    p2 = np.poly1d(z2)
#    print(z2)
#    print(p2) 
#    print 
#    
#    # 绘制曲线 
#    # 原曲线
#    pl.plot(x, y, 'b^-', label='Origin Line')
#    pl.plot(x, p1(x), 'gv--', label='Poly Fitting Line(deg=3)')
#    pl.plot(x, p2(x), 'r*', label='Poly Fitting Line(deg=6)')
#    pl.axis([0, 50, 0, 50])
#    pl.legend()
#    # Save figure
#    pl.savefig('scipy02.png', dpi=96)
#
#    exit()

    turns_and_sum = [
        h/float(w), 
        nonzero/float(total), 
        #horizontal_mean_sum, vertical_mean_sum, # train rate: 100.000000  test rate: 96.296296
        #horizontal_mean_mean, vertical_mean_mean, # train rate: 100.000000  test rate: 96.296296
        turn_horizontal_mean, turn_vertical_mean, # train rate: 100.000000  test rate: 97.037037
        #turn_row_mean, turn_col_mean, # train rate: 100.000000  test rate: 96.296296
        #turn_rows, turn_cols, # train rate: 100.000000  test rate: 96.666667
        row_turn_0, row_turn_1, row_turn_3, row_turn_5, row_turn_7, row_turn_9, row_turn_11, row_turn_15, row_turn_21, row_turn_more, # train rate: 98.888889  test rate: 97.407407
        col_turn_0, col_turn_1, col_turn_3, col_turn_5, col_turn_7, col_turn_9, col_turn_11, col_turn_15, col_turn_21, col_turn_more # train rate: 98.888889  test rate: 97.407407
    #] # train rate: 96.976568  test rate: 81.481481
    #] + hist # train rate: 100.000000  test rate: 94.028723
    #] + turn_each_row + turn_each_col + stroke_each_row + stroke_each_col + horizontal_sum.tolist() + vertical_sum.tolist() + distance_each_row + distance_each_col # train rate: 100.000000  test rate: 93.801965
    #] + turn_each_row + turn_each_col + stroke_each_row + stroke_each_col + horizontal_mean.tolist() + vertical_mean.tolist()  + distance_each_row + distance_each_col # train rate: 100.000000  test rate: 93.801965
    ] + turn_each_row + turn_each_col + stroke_each_row + stroke_each_col + horizontal_mean.tolist() + vertical_mean.tolist()  + distance_each_row + distance_each_col # train rate: 100.000000  test rate: 94.708995
    #] + hist + space_each_row + space_each_col + turn_each_row + turn_each_col + stroke_each_row + stroke_each_col + horizontal_mean.tolist() + vertical_mean.tolist()  + distance_each_row + distance_each_col # train rate: 100.000000  test rate: 94.255480
    #] + turn_each_row_fit.tolist() + turn_each_col_fit.tolist() + stroke_each_row_fit.tolist() + stroke_each_col_fit.tolist() + horizontal_sum_fit.tolist() + vertical_sum_fit.tolist() + distance_each_row_fit.tolist() + distance_each_col_fit.tolist() # train rate: 95.313681  test rate: 72.486772
    #] + stroke_each_row + stroke_each_col + turn_each_row + turn_each_col + horizontal_sum.tolist() + vertical_sum.tolist() + distance_each_row + distance_each_col # train rate: 100.000000  test rate: 93.801965
    #print turns_and_sum
    #exit()
    
    #turns_and_sum = [] + turn_each_row + turn_each_col + stroke_each_row + stroke_each_col + horizontal_sum.tolist() + vertical_sum.tolist() + distance_each_row + distance_each_col
    #return np.polyfit(np.arange(1, len(turns_and_sum) + 1, 1), np.array(turns_and_sum), 16) # train rate: 90.000000  test  rate: 50.000000

    #print len(turns_and_sum)
    #exit()

    return turns_and_sum # [round(i, 1) for i in turns_and_sum] # 



BLOCKS = 4
ORIENTATIONS = (8, 8, 4)
DIMENSION = 960

#print leargist.color_gist(Image.open(DATA_PATH + "/trains//simhei.且.png"), nblocks=BLOCKS, orientations=ORIENTATIONS).shape
#exit()

################## train ##########################

wd = DATA_PATH + "/trains/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

lsh = LSHash(6, DIMENSION)
print "extracting features ..."
t1 = time.time()
for i, fn in enumerate(files):
    #print unichr(ord(fn.split(".")[-2].decode("utf8")))
    #lsh.index(get_img(fn), extra_data=ord(fn.split(".")[-2].decode("utf8")))
    lsh.index(leargist.color_gist(Image.open(fn), nblocks=BLOCKS, orientations=ORIENTATIONS), extra_data=ord(fn.split(".")[-2].decode("utf8")))
t2 = time.time()
print "done. %d files took %0.3f ms" % (len(files), (t2 - t1) * 1000.0)

################## test ##########################

#rs = lsh.query(get_img(wd + '/simhei.且.png'), num_results=3, distance_func="euclidean")
#print [(unichr(r[0][1]), r[1]) for r in rs]
#rs = lsh.query(get_img(wd + '/simhei.且.png'), num_results=1, distance_func="euclidean")
#print unichr(rs[0][0][1])
#exit()

print "testing ..."

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
    rs = lsh.query(leargist.color_gist(Image.open(fn), nblocks=BLOCKS, orientations=ORIENTATIONS), num_results=1, distance_func="l1norm") 
    if rs and unichr(rs[0][0][1]) == fn.split(".")[-2].decode("utf8"):
        correct = correct + 1
t2 = time.time()
print "test rate: %f, %d files took %0.3f ms" % (correct/float(total)*100, len(files), (t2 - t1) * 1000.0)