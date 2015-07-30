#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image, ImageFont, ImageDraw
import os
import time

import pickle

import numpy as np
import matplotlib.pylab as pl

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

from skimage.feature import local_binary_pattern

import sys
sys.path.insert(1,'../img-processing/')
sys.path.insert(1,'../chinese-chars/')
sys.path.insert(1,'../orientation/')

from util import *
from parser import Hcl
from chinese_chars import *
from orientation import orientation

from lbp import ExtendedLBP, OriginalLBP, VarLBP, LPQ

COMPONENTS = 1000

DATA_PATH = "/mnt/hgfs/win/python"
#CHARS = primary_chars
#CHARS = primary_chars_1 + primary_chars_2 + primary_chars_3 + primary_chars_4
CHARS = [0x5176, 0x5171, 0x5177, 0x771F, 0x4E14, 0x76EE, 0x65E5, 0x6708, 0x66F0, 0x53BF, 0x672A, 0x672B, 0x6765, 0x4E4E, 0x5E73, 0x5DF1, 0x5DF2, 0x5DF3, 0x4E59, 0x98DE] #range(0x4E00, 0x9FA5): #0x9FFF
CHARS = CHARS + primary_chars_3[:10] + primary_chars_5[:10] + primary_chars_7[:10] + primary_chars_9[:10] + primary_chars_11[:10] + primary_chars_13[:10] + primary_chars_15[:10] + primary_chars_17[:10]
#其 : 0x5176
#共 : 0x5171
#具 : 0x5177
#真 : 0x771F
#且 : 0x4E14
#目 : 0x76EE
#日 : 0x65E5
#月 : 0x6708
#曰 : 0x66F0
#县 : 0x53BF
#未 : 0x672A
#末 : 0x672B
#来 : 0x6765
#乎 : 0x4E4E
#平 : 0x5E73
#己 : 0x5DF1
#已 : 0x5DF2
#巳 : 0x5DF3
#乙 : 0x4E59
#飞 : 0x98DE

def print_timing(func):
    def wrapper(*arg, **kwargs):
        t1 = time.time()
        res = func(*arg, **kwargs)
        t2 = time.time()
        print '%s took %0.3f ms' % (arg[0], (t2 - t1) * 1000.0)
        return res
    return wrapper

@print_timing
def generate_chars(font_path, folder, **kwargs):
    img_size = int(kwargs.get('img_size', 50))
    font_size = int(kwargs.get('font_size', 46))
    return_obj = bool(kwargs.get('return_obj', False))
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    font = ImageFont.truetype(font_path, font_size)

    for c in CHARS: 
        if type(c) is int:
            char = unichr(c)
        else:
            char = c
        (w, h) = font.getsize(char)
        offset = font.getoffset(char)
        img = Image.new('RGB', (img_size,img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text(((img_size - w - offset[0]) / 2, (img_size - h - offset[1]) / 2), char, font=font, fill=(0,0,0))
        if return_obj:
            return img
        #char_path = '%s/%s' % (folder, char)#hex(c)[2:])
        char_path = folder
        if not os.path.exists(char_path):
            os.makedirs(char_path)
        img.save('%s/%s.%s.png' % (char_path, font_name.decode("utf8"), char))

def generate_data(folder, filename):
    files = [fn for fn in os.listdir(folder)]
    files = [folder + fn for fn in files]

    fh = open(filename, 'w')

    for i in files:
        data = i.split(".")[-2].decode("utf8") + "," + ','.join(str(v) for v in get_img(i)) + "\n"
        fh.write(data.encode("utf8"))
    fh.close()

#    samples = []
#    responses = []
#    for i in files:
#        char = i.split(".")[-2].decode("utf8")
#        responses.append(char)
#        samples.append(get_img(i))
#
#    pca = RandomizedPCA(n_components=COMPONENTS)
#    std_scaler = StandardScaler()
#    samples = pca.fit_transform(samples)
#    samples = std_scaler.fit_transform(samples)
#
#    dat = ""
#    for i, sample in enumerate(samples):
#        char = responses[i]
#        dat += char + "," + ','.join(str(v) for v in sample) + "\n"
#
#    fh.write(dat.encode("utf8"))
#    fh.close()


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

def cal_distance(arr):
    nonzero = np.nonzero(arr)[0]
    if nonzero.any():
        return nonzero[-1] - nonzero[0]
    return 0 

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

def predict(img, pca, std_scaler, clf):
    x = pca.transform([img])
    x = std_scaler.transform(x)
    results = {"label": clf.predict(x)[0]}
    probs = {"prob_" + str(i) : prob for i, prob in enumerate(clf.predict_proba(x)[0])}
    results['probs'] = probs
    return results

def features(img):

    # settings for LBP
    METHOD = 'uniform'
    radius = 5
    n_points = 8 * radius
    #lbp = local_binary_pattern(img, n_points, radius, METHOD) # train rate: 72.108844  test rate: 30.914588
    #lbp = ExtendedLBP(radius=9)(img) # train rate: 99.622071  test rate: 76.643991
    #lbp = ExtendedLBP(radius=8)(img) # train rate: 99.773243  test rate: 78.155707
    #lbp = ExtendedLBP(radius=7)(img) # train rate: 96.825397  test rate: 76.719577
    #lbp = ExtendedLBP(radius=6)(img) # train rate: 100.000000  test rate: 83.673469
    #lbp = ExtendedLBP(radius=5)(img) # train rate: 99.848828  test rate: 87.906274
    #lbp = ExtendedLBP(radius=4)(img) # train rate: 99.546485  test rate: 87.981859
    #lbp = ExtendedLBP(radius=3)(img) # train rate: 91.836735  test rate: 78.080121
    #lbp = ExtendedLBP(radius=2)(img) # train rate: 92.970522  test rate: 75.510204
    #lbp = ExtendedLBP(radius=1)(img) # train rate: 78.004535  test rate: 60.846561
    #lbp = OriginalLBP()(img) # train rate: 84.278156  test rate: 63.794407
    #lbp = LPQ()(img) # train rate: 100.000000  test rate: 83.975813
    #lbp = LPQ(radius=1)(img) # train rate: 92.214664  test rate: 70.068027
    #lbp = LPQ(radius=2)(img) # train rate: 99.546485  test rate: 79.213908
    #lbp = LPQ(radius=3)(img) # train rate: 100.000000  test rate: 83.975813
    #lbp = LPQ(radius=4)(img) # train rate: 100.000000  test rate: 88.662132
    #lbp = LPQ(radius=5)(img) # train rate: 100.000000  test rate: 90.627362
    lbp = LPQ(radius=6)(img) # train rate: 100.000000  test rate: 90.400605
    #lbp = LPQ(radius=7)(img) # train rate: 100.000000  test rate: 90.627362
    #lbp = LPQ(radius=8)(img) # train rate: 100.000000  test rate: 89.720333
    #lbp = LPQ(radius=9)(img) # train rate: 100.000000  test rate: 87.150416
    #lbp = LPQ(radius=10)(img) # train rate: 100.000000  test rate: 86.772487

    n_bins = lbp.max() + 1
    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    #print len(hist.flatten().tolist())
    hist = hist.flatten().tolist()
    fillup = 256
    if len(hist) < fillup:
        hist = hist + ([0] * (fillup - len(hist)))
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
    ] + hist + turn_each_row + turn_each_col + stroke_each_row + stroke_each_col + horizontal_mean.tolist() + vertical_mean.tolist()  + distance_each_row + distance_each_col # train rate: 100.000000  test rate: 94.708995
    #] + hist + space_each_row + space_each_col + turn_each_row + turn_each_col + stroke_each_row + stroke_each_col + horizontal_mean.tolist() + vertical_mean.tolist()  + distance_each_row + distance_each_col # train rate: 100.000000  test rate: 94.255480
    #] + turn_each_row_fit.tolist() + turn_each_col_fit.tolist() + stroke_each_row_fit.tolist() + stroke_each_col_fit.tolist() + horizontal_sum_fit.tolist() + vertical_sum_fit.tolist() + distance_each_row_fit.tolist() + distance_each_col_fit.tolist() # train rate: 95.313681  test rate: 72.486772
    #] + stroke_each_row + stroke_each_col + turn_each_row + turn_each_col + horizontal_sum.tolist() + vertical_sum.tolist() + distance_each_row + distance_each_col # train rate: 100.000000  test rate: 93.801965
    #print turns_and_sum
    #exit()
    
    #turns_and_sum = [] + turn_each_row + turn_each_col + stroke_each_row + stroke_each_col + horizontal_sum.tolist() + vertical_sum.tolist() + distance_each_row + distance_each_col
    #return np.polyfit(np.arange(1, len(turns_and_sum) + 1, 1), np.array(turns_and_sum), 16) # train rate: 90.000000  test  rate: 50.000000

    return turns_and_sum # [round(i, 1) for i in turns_and_sum] # 


################### Generate HCL data ###########################

#parser = Hcl()
#parser.get_img('/mnt/hgfs/win/python/HCL2000/hh006.hcl', 2, 'test', mode='L')
#exit()

################### Generate features data ###########################

generate_data(DATA_PATH + "/trains/", DATA_PATH + '/features.data')
exit()

################### Generate train and test images ###########################

#/mnt/hgfs/win/python/fonts-train/simhei.ttf took 829.230 ms
#/mnt/hgfs/win/python/fonts-train/simsun.ttc took 583.972 ms
#/mnt/hgfs/win/python/fonts-train/simyou.ttf took 655.739 ms
#/mnt/hgfs/win/python/fonts-train/方正中倩简体.ttf took 483.509 ms
#/mnt/hgfs/win/python/fonts-train/方正中等线简体.ttf took 788.024 ms
#/mnt/hgfs/win/python/fonts-train/方正书宋简体.ttf took 1900.994 ms
#/mnt/hgfs/win/python/fonts-train/方正仿宋简体.ttf took 950.478 ms
#/mnt/hgfs/win/python/fonts-train/方正准圆简体.ttf took 754.321 ms
#/mnt/hgfs/win/python/fonts-train/方正北魏楷书简体.ttf took 4443.363 ms
#/mnt/hgfs/win/python/fonts-train/方正大标宋简体.ttf took 1259.127 ms
#/mnt/hgfs/win/python/fonts-train/方正大黑简体.ttf took 660.237 ms
#/mnt/hgfs/win/python/fonts-train/方正姚体简体.ttf took 1897.082 ms
#/mnt/hgfs/win/python/fonts-train/方正宋一简体.ttf took 939.088 ms
#/mnt/hgfs/win/python/fonts-train/方正宋三简体.ttf took 1057.673 ms
#/mnt/hgfs/win/python/fonts-train/方正宋黑简体.ttf took 1135.772 ms
#/mnt/hgfs/win/python/fonts-train/方正小标宋简体.ttf took 923.638 ms
#/mnt/hgfs/win/python/fonts-train/方正报宋简体.ttf took 2010.096 ms
#/mnt/hgfs/win/python/fonts-train/方正新报宋简体.ttf took 8306.070 ms
#/mnt/hgfs/win/python/fonts-train/方正楷体简体.ttf took 3677.246 ms
#/mnt/hgfs/win/python/fonts-train/方正粗倩简体.ttf took 2209.256 ms
#/mnt/hgfs/win/python/fonts-train/方正粗圆简体.ttf took 967.207 ms
#/mnt/hgfs/win/python/fonts-train/方正粗宋简体.ttf took 1013.340 ms
#/mnt/hgfs/win/python/fonts-train/方正细倩简体.ttf took 547.407 ms
#/mnt/hgfs/win/python/fonts-train/方正细圆简体.ttf took 835.610 ms
#/mnt/hgfs/win/python/fonts-train/方正细等线简体.ttf took 644.174 ms
#/mnt/hgfs/win/python/fonts-train/方正细黑一简体.ttf took 1281.633 ms
#/mnt/hgfs/win/python/fonts-train/方正黑体简体.TTF took 738.653 ms
#/mnt/hgfs/win/python/fonts-test/simhei.ttf took 1735.492 ms
#/mnt/hgfs/win/python/fonts-test/simkai.ttf took 2052.319 ms

#wd = DATA_PATH + "/fonts-train/"
#files = [fn for fn in os.listdir(wd)]
#files = [wd + fn for fn in files]
#
#for fn in files:
#    generate_chars(fn, DATA_PATH + '/trains/', img_size=50, font_size=46)
#
#wd = DATA_PATH + "/fonts-test/"
#files = [fn for fn in os.listdir(wd)]
#files = [wd + fn for fn in files]
#
#for fn in files:
#    generate_chars(fn, DATA_PATH + '/tests/', img_size=50, font_size=46)
#
#exit()


################### Analytic features ###########################
#wd = DATA_PATH + "/tests/"
#files = [fn for fn in os.listdir(wd)]
#files = [wd + fn for fn in files]
#
#for fn in files:
#    print fn
#    print get_img(fn)
#exit()
#
#fn = DATA_PATH + "/space.png"
#im_gray = cv2.imread(fn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
#(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#cv2.imwrite(DATA_PATH + "/space-rs.png", im_bw)
#print features(im_bw)
#exit()

#simhei.且.png : [0.9, 0.3, 14.3, 12.9, 0.3, 0.3, 8.0, 8.0, 3.2, 8.0, 3.7, 8.0, 2.0, 1.0, 9.0, 26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 16.0, 0.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#simkai.且.png : [0.8, 0.2, 8.5, 6.7, 0.2, 0.2, 20.0, 18.0, 3.8, 6.0, 3.2, 10.0, 0.0, 0.0, 5.0, 25.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 18.0, 5.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0]
#
#simhei.共.png : [1.0, 0.3, 12.9, 12.3, 0.3, 0.3, 25.0, 20.0, 3.3, 7.0, 4.7, 14.0, 4.0, 1.0, 4.0, 31.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.0, 10.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#simkai.共.png : [1.0, 0.2, 8.2, 7.8, 0.2, 0.2, 29.0, 28.0, 3.3, 11.0, 4.3, 15.0, 0.0, 3.0, 11.0, 24.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 11.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0]
#
#simhei.其.png : [1.0, 0.4, 14.5, 14.5, 0.4, 0.4, 21.0, 17.0, 3.2, 8.0, 5.7, 12.0, 4.0, 0.0, 8.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 6.0, 9.0, 10.0, 4.0, 0.0, 0.0, 0.0]
#simkai.其.png : [1.1, 0.2, 8.5, 8.9, 0.2, 0.2, 25.0, 35.0, 3.7, 17.0, 4.8, 18.0, 0.0, 2.0, 9.0, 25.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 6.0, 12.0, 5.0, 1.0, 1.0, 0.0, 0.0]
#
#simhei.具.png : [1.0, 0.4, 16.7, 15.9, 0.4, 0.4, 25.0, 18.0, 2.9, 10.0, 6.3, 10.0, 4.0, 0.0, 13.0, 22.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 9.0, 0.0, 7.0, 11.0, 0.0, 0.0, 0.0]
#simkai.具.png : [0.9, 0.2, 9.9, 9.1, 0.2, 0.2, 28.0, 28.0, 3.7, 10.0, 4.7, 15.0, 0.0, 3.0, 4.0, 27.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.0, 10.0, 3.0, 1.0, 5.0, 2.0, 0.0, 0.0]
#
#simhei.真.png : [1.0, 0.4, 17.1, 16.3, 0.4, 0.4, 24.0, 23.0, 2.6, 10.0, 8.3, 14.0, 3.0, 1.0, 20.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 12.0, 3.0, 2.0, 14.0, 0.0, 0.0]
#simkai.真.png : [1.0, 0.2, 8.7, 8.9, 0.2, 0.2, 25.0, 31.0, 3.5, 19.0, 5.7, 16.0, 0.0, 2.0, 13.0, 21.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 3.0, 8.0, 4.0, 4.0, 4.0, 0.0, 0.0]




################## train ##########################

wd = DATA_PATH + "/trains/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

data = []
labels = []
print "extracting features..."
for i, fn in enumerate(files):
    #print i, "of", len(files)
    data.append(get_img(fn))
    labels.append(fn.split(".")[-2].decode("utf8"))
print "done."

#pickle.dump(itemlist, outfile)
#itemlist = pickle.load(infile)

pca = RandomizedPCA(n_components=COMPONENTS)
#pca = RandomizedPCA(n_components=COMPONENTS)
#pca = RandomizedPCA()
std_scaler = StandardScaler()

#samples_train, samples_test, responses_train, responses_test = train_test_split(data, labels, test_size=0.1)
#
#print "scaling data..."
#samples_train = pca.fit_transform(samples_train)
#samples_test = pca.transform(samples_test)
#print "done."
#
#print "transforming data..."
#samples_train = std_scaler.fit_transform(samples_train)
#samples_test = std_scaler.transform(samples_test)
#print "done."
#
#print "training model..."
#clf = KNeighborsClassifier(n_neighbors=10)
#clf.fit(samples_train, responses_train)
#print "done"
#print "="*20
#print clf
#
#print "Confusion Matrix"
#print "="*40
#print confusion_matrix(responses_test, clf.predict(samples_test))

print "scaling data..."
data = pca.fit_transform(data)
print "done."

print "transforming data..."
data = std_scaler.fit_transform(data)
print "done."

print "training model..."
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(data, labels)
print "done"
print "="*20
print clf


####################### test ##########################

wd = DATA_PATH + "/tests/"
#files = [fn for fn in os.listdir(wd)] # all

files = [fn for fn in os.listdir(wd) if fn.find('simhei') != -1] # train
files = [wd + fn for fn in files]

correct = 0
total = 0
for i, fn in enumerate(files):
    total = total + 1
    #print predict(get_img(fn), pca, std_scaler, clf)['label'], fn.split(".")[-2].decode("utf8")
    if predict(get_img(fn), pca, std_scaler, clf)['label'] == fn.split(".")[-2].decode("utf8"):
        correct = correct + 1
print 'train rate: %f' % (correct/float(total)*100)

files = [fn for fn in os.listdir(wd) if fn.find('simkai') != -1] # test
files = [wd + fn for fn in files]

correct = 0
total = 0
for i, fn in enumerate(files):
    total = total + 1
    print predict(get_img(fn), pca, std_scaler, clf)['label'], fn.split(".")[-2].decode("utf8")
    if predict(get_img(fn), pca, std_scaler, clf)['label'] == fn.split(".")[-2].decode("utf8"):
        correct = correct + 1
print 'test  rate: %f' % (correct/float(total)*100)