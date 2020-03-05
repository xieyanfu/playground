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

    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

    #读入图像尺寸
    cols,rows,_=img.shape

    #缩放比例
    ratio=0.3

    #缩放后的尺寸
    cols=int(ratio*cols)
    rows=int(ratio*rows)

    #缩放图片
    img = cv2.resize(img,(rows,cols))
    cv2.imwrite('resize-' + sys.argv[2], img)

    #获得三个通道
    Bch,Gch,Rch=cv2.split(img) 

    # cv2.imshow('Blue channel',cv2.merge([Bch,0*Gch,0*Rch]))
    # cv2.imshow('Green channel',cv2.merge([0*Bch,Gch,0*Rch]))
    # cv2.imshow('Red channel',cv2.merge([0*Bch,0*Gch,Rch]))
    # 
    # cv2.imshow('Blue channel',Bch)
    # cv2.imshow('Green channel',Gch)
    # cv2.imshow('Red channel',Rch)

    #红色通道阈值
    avg = (int(np.amin(Rch)) + int(np.amax(Rch))) / 5
    _,RedThresh = cv2.threshold(Rch,avg,255,cv2.THRESH_BINARY)
    _,RedThreshInv = cv2.threshold(Rch,avg,255,cv2.THRESH_BINARY_INV)

    # #膨胀操作
    # element = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)) 
    # erode = cv2.erode(RedThresh, element)
    # cv2.imshow("erode",erode) 

    # erosion = util.util('erosion', RedThresh)
    # cv2.imshow("erosion",erosion) 

    # dilation = util.util('dilation', erosion)
    # cv2.imshow("dilation",dilation) 

    
    #膨胀操作
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) 
    mask = cv2.dilate(RedThreshInv, element, 5) # background black
    # cv2.imshow("mask",mask) 

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 1)
    # cv2.imshow("opening",opening) 
    # sure background area
    mask = cv2.dilate(opening,kernel,iterations=1)
    # cv2.imshow("mask2",mask) 

    # change char to black
    mask = (mask ^ 1) * 255
    # cv2.imshow("mask3",mask) 



    Bch[mask == 255] = 255
    # cv2.imshow("Bch", Bch) 

    Gch[mask == 255] = 255
    # cv2.imshow("Gch", Gch) 

    Rch[mask == 255] = 255
    # cv2.imshow("Rch", Rch) 
    
    
    merged = cv2.merge([Bch,Gch,Rch])

    mergedGray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    avg = (int(np.amin(mergedGray)) + int(np.amax(mergedGray))) / 3
    _,mergedThresh = cv2.threshold(mergedGray,avg,255,cv2.THRESH_BINARY)


    _ ,contours, hierarchy = cv2.findContours(mergedThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    temp = np.zeros(mergedThresh.shape)
    temp.fill(255)
    cv2.drawContours(temp, contours, -1, (0,0,0), 1) 


    #显示效果
    # cv2.imshow('original color image', img)
    # cv2.imshow("RedThresh",RedThresh) 
    # cv2.imshow("RedThreshInv",RedThreshInv) 

    cv2.imshow("Original", img) 
    cv2.imshow("Merged", merged) 
    cv2.imshow("mergedThresh", mergedThresh) 
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
    
    



