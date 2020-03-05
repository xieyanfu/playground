#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path
import sys  

import cv2
import numpy as np
import util

# ===============================================================================

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print("Usage:", sys.argv[0], " inputImg outputImg")
        exit()
        
    if not os.path.isfile(sys.argv[1]):
        print(sys.argv[1], " doesn't exist")
        exit()

    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

    #读入图像尺寸
    cols,rows=img.shape

    #缩放比例
    ratio=0.3

    #缩放后的尺寸
    cols=int(ratio*cols)
    rows=int(ratio*rows)

    #缩放图片
    img = cv2.resize(img,(rows,cols))
    cv2.imwrite('resize-' + sys.argv[2], img)

    img = util.util('binarize', img)
    # img = util.util('dilation', img)
    # img = util.util('binarize', img)

    kernel = np.ones((3,3),np.uint8)
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations = 1)
    # cv2.imshow("opening",opening) 
    # sure background area
    img = cv2.dilate(img,kernel,iterations=1)

    _ ,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    temp = np.zeros(img.shape)
    temp.fill(255)
    cv2.drawContours(temp, contours, -1, (0,0,0), 1) 

    cv2.imshow("Original", img) 
    cv2.imshow("contours", temp) 

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    exit()

    # img = util.util('noise_removal', img)
    # img = util.util('sharpening', img)

    # Ä«¼£
    img = util.util('binarize', img)
    cv2.imwrite("binarize-" + sys.argv[1], img)

    img = util.util('erosion', img)
    cv2.imwrite("erosion-" + sys.argv[1], img)

    img = util.util('dilation', img)
    cv2.imwrite("dilation-" + sys.argv[1], img)
    
    # img = util.util('noise_removal', img)
    #img = util.util('crop', img)
    #img = util.util('resize', img)
    #img = util.util('thin_zhangsuen', img)
    #img = util.util('frame', img)
    
    temp = np.zeros(img.shape, np.uint8)
    temp.fill(255)
    
    _ ,contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(temp, contours, -1, (0,0,0), 1) 

    img = temp

    cv2.imwrite(sys.argv[2], img)
    
    



