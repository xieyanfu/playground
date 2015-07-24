#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image, ImageFont, ImageDraw
import os
import time

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

import sys
sys.path.insert(1,'../img-processing/')

from util import *

DATA_PATH = "/mnt/hgfs/win/python"

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
    font_name = os.path.splitext(os.path.basename(font_path))[0]
    font = ImageFont.truetype(font_path, font_size)
    #其 : 0x5176
    #共 : 0x5171
    #具 : 0x5177
    #真 : 0x771F
    #且 : 0x4E14
    for c in (0x5176, 0x5171, 0x5177, 0x771F, 0x4E14): #range(0x4E00, 0x9FA5): #0x9FFF
        char = unichr(c)
        (w, h) = font.getsize(char)
        offset = font.getoffset(char)
        img = Image.new('RGB', (img_size,img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text(((img_size - w - offset[0]) / 2, (img_size - h - offset[1]) / 2), char, font=font, fill=(0,0,0))
        #char_path = '%s/%s' % (folder, char)#hex(c)[2:])
        char_path = folder
        if not os.path.exists(char_path):
            os.makedirs(char_path)
        img.save('%s/%s.%s.png' % (char_path, font_name.decode("utf8"), char))

def count_turn(arr):
    turn = 0;
    base = arr[0]
    for i in arr[1:]:
        if i != base:
            turn = turn + 1
            base = i
    return turn

def get_img(img):
    im_gray = cv2.imread(fn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return features(im_bw)

def predict(img, pca, std_scaler, clf):
    x = pca.transform([img])
    x = std_scaler.transform(x)
    results = {"label": clf.predict(x)[0]}
    probs = {"prob_" + str(i) : prob for i, prob in enumerate(clf.predict_proba(x)[0])}
    results['probs'] = probs
    return results

def features(img):
    # ((img ^ 1) * 255)
    img = (img != 255) * 1
    #print img
    #print 
    nonzero = np.nonzero(img)
    img = img[min(nonzero[0]):max(nonzero[0])+1,min(nonzero[1]):max(nonzero[1])+1]
    rows = np.vsplit(img, img.shape[0])
    cols = np.hsplit(img, img.shape[1])
    #print img
    #print 

#    for i in xrange(img.shape[0]):
#        row = ""
#        for j in xrange(img.shape[1]):
#            row = row + ("#" if img[i,j] == 1 else "-")
#        print row

    h = img.shape[0]
    w = img.shape[1]
    total = img.shape[0]*img.shape[1]
    nonzero = np.count_nonzero(img)
    #print h, w, total, nonzero

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

    return [round(i, 2) for i in [
        h, w, total, nonzero, 
        horizontal_min_sum, horizontal_mean_sum, horizontal_max_sum, vertical_min_sum, vertical_mean_sum, vertical_max_sum, 
        horizontal_min_mean, horizontal_mean_mean, horizontal_max_mean, vertical_min_mean, vertical_mean_mean, vertical_max_mean,
        turn_horizontal_min, turn_horizontal_mean, turn_horizontal_max, turn_vertical_min, turn_vertical_mean, turn_vertical_max,
        turn_row_mean, turn_rows, turn_col_mean, turn_cols
    ]]

################### Generate train and test images ###########################

wd = DATA_PATH + "/fonts-train/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

#for fn in files:
#    generate_chars(fn, DATA_PATH + '/trains/', img_size=50, font_size=46)

wd = DATA_PATH + "/fonts-test/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

#for fn in files:
#    generate_chars(fn, DATA_PATH + '/tests/', img_size=50, font_size=46)



################### Analytic features ###########################
#wd = DATA_PATH + "/tests/"
#files = [fn for fn in os.listdir(wd)]
#files = [wd + fn for fn in files]
#
#for fn in files:
#    print fn
#    print get_img(fn)


#im_gray = cv2.imread(fn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
#(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#cv2.imwrite(DATA_PATH + "/space-rs.png", im_bw)
#print im_bw
#im_bw = util("thin_zhangsuen", im_bw)
#print features(im_bw)


#simhei.且.png : [13.0, 14.0, 182.0, 63.0, 2.0, 4.85, 14.0, 1.0, 4.5, 13.0, 0.14, 0.35, 1.0, 0.08, 0.35, 1.0, 4.0, 5.0, 0.0, 1.0, 7.0, 0.0, 3.08, 6.0, 3.0, 4.0]
#simkai.且.png : [12.0, 14.0, 168.0, 48.0, 0.0, 4.0, 14.0, 0.0, 3.43, 12.0, 0.0, 0.29, 1.0, 0.0, 0.29, 1.0, 0.0, 8.0, 0.0, 0.0, 9.0, 0.0, 3.25, 3.0, 3.0, 6.0]
#
#simhei.共.png : [13.0, 14.0, 182.0, 79.0, 0.0, 6.08, 14.0, 2.0, 5.64, 13.0, 0.0, 0.43, 1.0, 0.15, 0.43, 1.0, 0.0, 8.0, 0.0, 2.0, 8.0, 0.0, 3.08, 5.0, 4.14, 10.0]
#simkai.共.png : [13.0, 14.0, 182.0, 50.0, 0.0, 3.85, 14.0, 0.0, 3.57, 13.0, 0.0, 0.27, 1.0, 0.0, 0.27, 1.0, 0.0, 11.0, 0.0, 0.0, 11.0, 0.0, 3.23, 7.0, 4.0, 8.0]
#
#simhei.其.png : [15.0, 14.0, 210.0, 98.0, 0.0, 6.53, 14.0, 2.0, 7.0, 15.0, 0.0, 0.47, 1.0, 0.13, 0.47, 1.0, 0.0, 8.0, 0.0, 2.0, 11.0, 0.0, 2.8, 9.0, 4.5, 7.0]
#simkai.其.png : [14.0, 14.0, 196.0, 54.0, 0.0, 3.86, 14.0, 0.0, 3.86, 14.0, 0.0, 0.28, 1.0, 0.0, 0.28, 1.0, 0.0, 9.0, 0.0, 0.0, 11.0, 0.0, 3.0, 7.0, 3.79, 10.0]
#
#
#simhei.具.png : [13.0, 14.0, 182.0, 108.0, 0.0, 8.31, 14.0, 2.0, 7.71, 13.0, 0.0, 0.59, 1.0, 0.15, 0.59, 1.0, 0.0, 9.0, 0.0, 2.0, 10.0, 0.0, 2.54, 9.0, 4.21, 7.0]
#simkai.具.png : [13.0, 13.0, 169.0, 53.0, 0.0, 4.08, 13.0, 0.0, 4.08, 13.0, 0.0, 0.31, 1.0, 0.0, 0.31, 1.0, 0.0, 11.0, 0.0, 0.0, 9.0, 0.0, 3.38, 3.0, 4.23, 7.0]
#
#simhei.真.png : [14.0, 14.0, 196.0, 114.0, 0.0, 8.14, 14.0, 2.0, 8.14, 14.0, 0.0, 0.58, 1.0, 0.14, 0.58, 1.0, 0.0, 11.0, 0.0, 2.0, 10.0, 0.0, 2.57, 8.0, 5.07, 9.0]
#simkai.真.png : [14.0, 14.0, 196.0, 63.0, 0.0, 4.5, 14.0, 0.0, 4.5, 14.0, 0.0, 0.32, 1.0, 0.0, 0.32, 1.0, 0.0, 9.0, 0.0, 0.0, 11.0, 0.0, 2.86, 7.0, 3.64, 9.0]




################## train ##########################

wd = DATA_PATH + "/trains/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

data = []
labels = []
print "extracting features..."
for i, fn in enumerate(files):
    #print i, "of", len(files)
    #data.append(get_image_data(fn))
    data.append(get_img(fn))
    labels.append(fn.split(".")[-2].decode("utf8"))
print "done."

pca = RandomizedPCA(n_components=5)
std_scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
#print X_train, X_test, y_train, y_test

print "scaling data..."
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print "done."

print "transforming data..."
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)
print "done."

print "training model..."
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_train, y_train)
print "done"
print "="*20
print clf

print "Confusion Matrix"
print "="*40
print confusion_matrix(y_test, clf.predict(X_test))



####################### test ##########################

wd = DATA_PATH + "/tests/"
#files = [fn for fn in os.listdir(wd)] # all

files = [fn for fn in os.listdir(wd) if fn.find('simhei') != -1] # train
files = [wd + fn for fn in files]

correct = 0
total = 0
for i, fn in enumerate(files):
    total = total + 1
    print predict(get_img(fn), pca, std_scaler, clf)['label'], fn.split(".")[-2].decode("utf8")
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