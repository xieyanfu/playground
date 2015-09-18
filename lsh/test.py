#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import sys

sys.path.insert(1,'../../LSHash/')
sys.path.insert(1,'../chinese-chars/')

from chinese_chars import *

from PIL import Image
from lshash import LSHash
import leargist



CHARS = primary_chars_1 + primary_chars_2[:4] + primary_chars_3[:4] + primary_chars_4[:5] + primary_chars_5[:5] + primary_chars_6[:5] + primary_chars_7[:5] + primary_chars_8[:5] + primary_chars_9[:5] + primary_chars_10[:5] + primary_chars_11[:5] + primary_chars_12[:5] + primary_chars_13[:5] + primary_chars_14[:5] + primary_chars_15[:5] + primary_chars_16[:5] + primary_chars_17[:5] + primary_chars_18[:5] + primary_chars_19[:5] + primary_chars_20[:5] + primary_chars_21[:5] + primary_chars_22 + primary_chars_23



BLOCKS = 5
ORIENTATIONS = (8,)
DISTANCE_FUNC = "cosine" # l1norm, cosine, euclidean, true_euclidean, centred_euclidean

folder = "/home/www/ocr/trains"
files = [os.path.join(folder,fn) for fn in os.listdir(folder)]

DIMENSION = leargist.color_gist(Image.open(files[0]), nblocks=BLOCKS, orientations=ORIENTATIONS).shape[0]


# Blocks: 5 , Orientations: (8,), Dimension: 600, DISTANCE_FUNC: cosine             test rate: 94.148148, 1350 files took 281909.610 ms
# Blocks: 5 , Orientations: (8,), Dimension: 600, DISTANCE_FUNC: l1norm             test rate: 91.407407, 1350 files took 849482.719 ms
# Blocks: 5 , Orientations: (8,), Dimension: 600, DISTANCE_FUNC: euclidean          test rate: 93.777778, 1350 files took 180138.969 ms
# Blocks: 5 , Orientations: (8,), Dimension: 600, DISTANCE_FUNC: true_euclidean     test rate: 93.777778, 1350 files took 257876.009 ms
# Blocks: 5 , Orientations: (8,), Dimension: 600, DISTANCE_FUNC: centred_euclidean  test rate: 5.851852,  1350 files took 374195.665 ms




################## train ##########################

samples = []
responses = []

print 'Blocks: %d , Orientations: %s, Dimension: %d, DISTANCE_FUNC: %s' % (BLOCKS, ORIENTATIONS, DIMENSION, DISTANCE_FUNC)

print 'extracting features ...'
t1 = time.time()
for i in files:
    responses.append(CHARS.index(i.split(".")[-2].decode("utf8")) + 1)
    samples.append(leargist.color_gist(Image.open(i), nblocks=BLOCKS, orientations=ORIENTATIONS))
t2 = time.time()
print 'done, %d file took %0.3f ms' % (len(files), (t2 - t1) * 1000.0)


train_n = int(len(files)*0.5)

lsh = LSHash(3, DIMENSION, num_hashtables=5)
print "indexing ..."
t1 = time.time()
for i, sample in enumerate(samples[:train_n]):
    lsh.index(sample, extra_data=responses[:train_n][i])
t2 = time.time()
print "done. %d files took %0.3f ms" % (train_n, (t2 - t1) * 1000.0)




################## test ##########################

print "testing ..."

#correct = 0
#total = 0
#t1 = time.time()
#for i, sample in enumerate(samples[:train_n]):
#    total = total + 1
#    rs = lsh.query(sample, num_results=3, distance_func=DISTANCE_FUNC) 
#    if rs and rs[0][0][1] == responses[:train_n][i]:
#        correct = correct + 1
##    if rs:
##        rs = [r[0][1] for r in rs]
##        try:
##            idx = rs.index(responses[:train_n][i])
##        except ValueError:
##            idx = -1
##        if idx != -1:
##            correct = correct + 1
#    #else:
#    #    print CHARS[rs[0][0][1]], " => ", CHARS[responses[:train_n][i]]
#t2 = time.time()
#print "train rate: %f, %d files took %0.3f ms" % (correct/float(total)*100, total, (t2 - t1) * 1000.0)

correct = 0
total = 0
t1 = time.time()
for i, sample in enumerate(samples[train_n:]):
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
    rs = lsh.query(sample, num_results=1, distance_func=DISTANCE_FUNC) 
    if rs and rs[0][0][1] == responses[train_n:][i]:
        correct = correct + 1
#    if rs:
#        rs = [r[0][1] for r in rs]
#        try:
#            idx = rs.index(responses[train_n:][i])
#        except ValueError:
#            idx = -1
#        if idx != -1:
#            correct = correct + 1
    #else:
    #    print CHARS[rs[0][0][1]], " => ", CHARS[responses[train_n:][i]]

t2 = time.time()
print "test rate: %f, %d files took %0.3f ms" % (correct/float(total)*100, total, (t2 - t1) * 1000.0)