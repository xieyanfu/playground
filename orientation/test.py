#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import Image, ImageFont, ImageDraw
from StringIO import StringIO
import base64
import os
import time
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

KSIZE = 10
SMOOTH = False

STANDARD_SIZE = (50, 50)

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

f = lambda x,y: 2*x*y
g = lambda x,y: x**2 - y**2
def orientation(img, block_size, smooth=False):
    h, w = img.shape

    # make a reflect border frame to simplify kernel operation on borders
    borderedImg = cv2.copyMakeBorder(img, block_size,block_size,block_size,block_size, cv2.BORDER_DEFAULT)

    # apply a gradient in both axis
    sobelx = cv2.Sobel(borderedImg, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(borderedImg, cv2.CV_64F, 0, 1, ksize=3)

    angles = np.zeros((h/block_size, w/block_size), np.float32)

    for i in xrange(w/block_size):
        for j in xrange(h/block_size):
            nominator = 0.
            denominator = 0.

            # calculate the summation of nominator (2*Gx*Gy)
            # and denominator (Gx^2 - Gy^2), where Gx and Gy
            # are the gradient values in the position (j, i)
            for k in xrange(block_size):
                for l in xrange(block_size):
                    posX = block_size-1 + (i*block_size) + k
                    posY = block_size-1 + (j*block_size) + l
                    valX = sobelx.item(posY, posX)
                    valY = sobely.item(posY, posX)

                    nominator += f(valX, valY)
                    denominator += g(valX, valY)
            
            # if the strength (norm) of the vector 
            # is not greater than a threshold
            if math.sqrt(nominator**2 + denominator**2) < 1000000:
                angle = 0.
            else:
                if denominator >= 0:
                    angle = cv2.fastAtan2(nominator, denominator)
                elif denominator < 0 and nominator >= 0:
                    angle = cv2.fastAtan2(nominator, denominator) + math.pi
                else:
                    angle = cv2.fastAtan2(nominator, denominator) - math.pi
                angle /= float(2)

            angles.itemset((j, i), angle)
    
    if smooth:
        angles = cv2.GaussianBlur(angles, (3,3), 0, 0)
    return angles

def get_img(img, block_size=5, smooth=False):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ori = orientation(gray, 5, smooth)
    return ori.reshape(1, ori.shape[0] * ori.shape[1])[0]

def get_image_data(filename):
    img = Image.open(filename)
    img = img.getdata()
    img = img.resize(STANDARD_SIZE)
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

def predict(img, pca, std_scaler, clf):
    x = pca.transform([img])
    x = std_scaler.transform(x)
    results = {"label": clf.predict(x)[0]}
    probs = {"prob_" + str(i) : prob for i, prob in enumerate(clf.predict_proba(x)[0])}
    results['probs'] = probs
    return results

################### Generate train and test images ###########################

wd = DATA_PATH + "/fonts-train/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

for fn in files:
    generate_chars(fn, 'trains/')

wd = DATA_PATH + "/fonts-test/"
files = [fn for fn in os.listdir(wd)]
files = [wd + fn for fn in files]

for fn in files:
    generate_chars(fn, 'tests/')



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
    data.append(get_img(fn, block_size=KSIZE, smooth=SMOOTH))
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
    #print predict(get_image_data(fn), pca, std_scaler, clf)['label'], fn.split(".")[-2].decode("utf8")
    #if predict(get_image_data(fn), pca, std_scaler, clf)['label'] == fn.split(".")[-2].decode("utf8"):
    print predict(get_img(fn, block_size=KSIZE, smooth=SMOOTH), pca, std_scaler, clf)['label'], fn.split(".")[-2].decode("utf8")
    if predict(get_img(fn, block_size=KSIZE, smooth=SMOOTH), pca, std_scaler, clf)['label'] == fn.split(".")[-2].decode("utf8"):
        correct = correct + 1
print 'train rate: %f' % (correct/float(total)*100)

files = [fn for fn in os.listdir(wd) if fn.find('simkai') != -1] # test
files = [wd + fn for fn in files]

correct = 0
total = 0
for i, fn in enumerate(files):
    total = total + 1
    #print predict(get_image_data(fn), pca, std_scaler, clf)['label'], fn.split(".")[-2].decode("utf8")
    #if predict(get_image_data(fn), pca, std_scaler, clf)['label'] == fn.split(".")[-2].decode("utf8"):
    print predict(get_img(fn, block_size=KSIZE, smooth=SMOOTH), pca, std_scaler, clf)['label'], fn.split(".")[-2].decode("utf8")
    if predict(get_img(fn, block_size=KSIZE, smooth=SMOOTH), pca, std_scaler, clf)['label'] == fn.split(".")[-2].decode("utf8"):
        correct = correct + 1
print 'test  rate: %f' % (correct/float(total)*100)