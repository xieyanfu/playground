#!/usr/bin/python
# -*- coding: utf-8 -*-

# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_ml/py_knn/py_knn_opencv/py_knn_opencv.html#knn-opencv

import numpy as np
import cv2

import os, sys

# realpath() will make your script run, even if you symlink it :)
ext = os.path.realpath(os.path.abspath('../chinese-chars'))
if ext not in sys.path:
    sys.path.insert(0, ext)

from chinese_chars import primary_chars

# train data
train_data = []
train_lablel = []
for c in primary_chars:
    for f in ['simsun']:
        for s in ['12']:
            p = 'train/%s/%s-%s.png' % (c, f, s)
            img = cv2.imread(p.encode('utf-8'), cv2.CV_LOAD_IMAGE_GRAYSCALE)
            train_data.append(img.reshape(-1))
            train_lablel.append(ord(c))

# test data
test_pattern = 'train/%s/simsun-12.png' # same small size
#test_pattern = 'test/%s/simsun-14.png' # diff small size
#test_pattern = 'test/%s/simsun-32.png' # same large size

test_data = []
test_lablel = []
for c in primary_chars:
    p = test_pattern % c
    img = cv2.imread(p.encode('utf-8'), cv2.CV_LOAD_IMAGE_GRAYSCALE)
    test_data.append(img.reshape(-1))
    test_lablel.append(ord(c))


# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.KNearest()
knn.train(np.array(train_data, dtype = np.float32), np.array(train_lablel, dtype = np.float32))
ret,result,neighbours,dist = knn.find_nearest(np.array(test_data, dtype = np.float32), k=5)

result = result.reshape(result.shape[0] * result.shape[1])

print result
print test_lablel

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_lablel
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy


## save the data
#np.savez('knn_data.npz',train=train, train_labels=train_labels)
#
## Now load the data
#with np.load('knn_data.npz') as data:
#    print data.files
#    train = data['train']
#    train_labels = data['train_labels']

