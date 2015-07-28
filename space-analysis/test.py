#!/usr/bin/python
# -*- coding: utf-8 -*-

import Image, ImageFont, ImageDraw
import os
import time

import pickle

from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

import sys
sys.path.insert(1,'../img-processing/')
sys.path.insert(1,'../chinese-chars/')
sys.path.insert(1,'../orientation/')

from util import *
from chinese_chars import *
from orientation import orientation

DATA_PATH = "/mnt/hgfs/win/python"
#CHARS = primary_chars
#CHARS = primary_chars_1 + primary_chars_2 + primary_chars_3 + primary_chars_4
CHARS = (0x5176, 0x5171, 0x5177, 0x771F, 0x4E14, 0x76EE, 0x65E5, 0x6708, 0x66F0, 0x53BF) #range(0x4E00, 0x9FA5): #0x9FFF
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

    samples = []
    responses = []
    for i in files:
        char = i.split(".")[-2].decode("utf8")
        responses.append(char)
        samples.append(get_img(i))

    pca = RandomizedPCA(n_components=10)
    std_scaler = StandardScaler()
    samples = pca.fit_transform(samples)
    samples = std_scaler.fit_transform(samples)

    dat = ""
    for i, sample in enumerate(samples):
        char = responses[i]
        dat += char + "," + ','.join(str(v) for v in sample) + "\n"

    fh = open(filename, 'w')
    fh.write(dat.encode("utf8"))
    fh.close()

def count_turn(arr):
    turn = 0;
    base = arr[0]
    for i in arr[1:]:
        if i != base:
            turn = turn + 1
            base = i
    return turn

def get_img(fn):
    im_gray = cv2.imread(fn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    angles = orientation(im_gray, 10, False)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return features(im_bw)# + angles.flatten().tolist()

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

    turns_and_sum = [
        h/float(w), 
        #nonzero/float(total), 
        #horizontal_mean_sum, vertical_mean_sum, 
        #horizontal_mean_mean, vertical_mean_mean, 
        #turn_horizontal_mean, turn_vertical_mean, 
        #turn_row_mean, turn_rows, turn_col_mean, turn_cols,
        row_turn_0, row_turn_1, row_turn_3, row_turn_5, row_turn_7, row_turn_9, row_turn_11, row_turn_15, row_turn_21, row_turn_more,
        col_turn_0, col_turn_1, col_turn_3, col_turn_5, col_turn_7, col_turn_9, col_turn_11, col_turn_15, col_turn_21, col_turn_more
    ] #+ turn_each_row + turn_each_col + horizontal_sum.tolist() + vertical_sum.tolist()
    return [round(i, 1) for i in turns_and_sum] # round(i, 1)



################### Generate features data ###########################

#generate_data(DATA_PATH + "/trains/", DATA_PATH + '/features.data')
#exit()

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

pca = RandomizedPCA(n_components=10)
#pca = RandomizedPCA(n_components=100)
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
    #print predict(get_img(fn), pca, std_scaler, clf)['label'], fn.split(".")[-2].decode("utf8")
    if predict(get_img(fn), pca, std_scaler, clf)['label'] == fn.split(".")[-2].decode("utf8"):
        correct = correct + 1
print 'test  rate: %f' % (correct/float(total)*100)