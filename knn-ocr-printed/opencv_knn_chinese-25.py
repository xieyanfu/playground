#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

import os, sys

# realpath() will make your script run, even if you symlink it :)
ext = os.path.realpath(os.path.abspath('../chinese-chars'))
if ext not in sys.path:
    sys.path.insert(0, ext)

from chinese_chars import primary_chars

# np.set_printoptions(threshold='nan')

img = cv2.imread('chinese-25.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 8000 cells, each 30x30 size
cells = [np.hsplit(row,8) for row in np.vsplit(gray,1000)]

# Make it into a Numpy array. It size will be (1000, 8, 30, 30)
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:].reshape(-1,900).astype(np.float32) # Size = (8000, 900)
test = x[:,:].reshape(-1,900).astype(np.float32) # Size = (8000, 900)

# Create labels for train and test data
k = [ord(c) for c in primary_chars[:1000]]
train_labels = np.repeat(k,8)[:,np.newaxis]
test_labels = train_labels.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv2.KNearest()
knn.train(train,train_labels)
ret,result,neighbours,dist = knn.find_nearest(test,k=5)

# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
matches = result==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100.0/result.size
print accuracy