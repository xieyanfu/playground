#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import cv2
import numpy as np
from scipy import ndimage
import weave
from PIL import Image

class BaseThin:
    def get_img(self, img):
        src = cv2.imread(img)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, banary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        return banary / 255 ^ 1

    def save_img(self, filename, data):
        data ^= 1
        cv2.imwrite(filename, data * 255)

class ZhangSuen2(BaseThin):

    def _neighbours(self, x, y, image):
        '''Return 8-neighbours of point p1 of picture, in order'''
        i = image
        x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1
        #print ((x,y))
        return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5
                i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9

    def _transitions(self, neighbours):
        n = neighbours + neighbours[0:1]    # P2, ... P9, P2
        return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

    def thin(self, image):
        changing1 = changing2 = [(-1, -1)]
        
        counter = 0
        
        while changing1 or changing2:
            counter = counter + 1
            
            if counter > 20:
                break
            # Step 1
            changing1 = []
            for y in range(1, len(image) - 1):
                for x in range(1, len(image[0]) - 1):
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = self._neighbours(x, y, image)
                    if (image[y][x] == 1 and    # (Condition 0)
                        P4 * P6 * P8 == 0 and   # Condition 4
                        P2 * P4 * P6 == 0 and   # Condition 3
                        self._transitions(n) == 1 and # Condition 2
                        2 <= sum(n) <= 6):      # Condition 1
                        changing1.append((x,y))
                        #print(x,y)
            for x, y in changing1: image[y][x] = 0
            # Step 2
            # print('step2')
            changing2 = []
            for y in range(1, len(image) - 1):
                for x in range(1, len(image[0]) - 1):
                    P2,P3,P4,P5,P6,P7,P8,P9 = n = self._neighbours(x, y, image)
                    if (image[y][x] == 1 and    # (Condition 0)
                        P2 * P6 * P8 == 0 and   # Condition 4
                        P2 * P4 * P8 == 0 and   # Condition 3
                        self._transitions(n) == 1 and # Condition 2
                        2 <= sum(n) <= 6):      # Condition 1
                        changing2.append((x,y))
                        
                       # print(x,y)
                        
            for x, y in changing2: image[y][x] = 0
            #print changing1
            #print changing2
        return image

class ZhangSuen(BaseThin):
    """
    This is the implementation of the Zhang-Suen thinning algorithm using OpenCV. 
    The algorithm is explained in "A fast parallel algorithm for thinning digital 
    patterns" by T.Y. Zhang and C.Y. Suen.

    http://opencv-code.com/quick-tips/implementation-of-thinning-algorithm-in-opencv/
    https://github.com/bsdnoobz/zhang-suen-thinning
    """
    def _thinningIteration(self, im, iter):
        I, M = im, np.zeros(im.shape, np.uint8)
        expr = """
        for (int i = 1; i < NI[0]-1; i++) {
            for (int j = 1; j < NI[1]-1; j++) {
                int p2 = I2(i-1, j);
                int p3 = I2(i-1, j+1);
                int p4 = I2(i, j+1);
                int p5 = I2(i+1, j+1);
                int p6 = I2(i+1, j);
                int p7 = I2(i+1, j-1);
                int p8 = I2(i, j-1);
                int p9 = I2(i-1, j-1);

                int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
                    M2(i,j) = 1;
                }
            }
        } 
        """

        weave.inline(expr, ["I", "iter", "M"])
        return (I & ~M)

    def thin(self, img):
        prev = np.zeros(img.shape[:2], np.uint8)
        diff = None

        while True:
            img = self._thinningIteration(img, 0)
            img = self._thinningIteration(img, 1)
            diff = np.absolute(img - prev)
            prev = img.copy()
            if np.sum(diff) == 0:
                break

        return img

class Morphological(BaseThin):
    """
    一种保形的数学形态学图像细化算法
    """
    def _thin(self, image, trim_iteration=0):
    
        d_hit = np.array([[0, 1, 0],
                        [0, 1, 1],
                        [0, 0, 0]])
        d_miss = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0]])
        
        e_hit = np.array([[0, 0, 1],
                        [0, 1, 1],
                        [0, 0, 1]])
        e_miss = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0]])
        
        k_hit = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])
        k_miss = np.array([[1, 1, 0],
                        [1, 0, 0],
                        [1, 1, 0]])
        
        d_hit_list = [d_hit]
        d_miss_list = [d_miss]
        
        e_hit_list = [e_hit]
        e_miss_list = [e_miss]
        
        k_hit_list = [k_hit]
        k_miss_list = [k_miss]
        
        for i in range(3):
            d_hit_list.insert(0, np.rot90(d_hit, i + 1))
            d_miss_list.insert(0, np.rot90(d_miss, i + 1))
            
            e_hit_list.insert(0, np.rot90(e_hit, i + 1))
            e_miss_list.insert(0, np.rot90(e_miss, i + 1))
            
            k_hit_list.insert(0, k_hit)
            k_miss_list.insert(0, np.rot90(k_miss, i + 1))
        
        d = zip(d_hit_list, d_miss_list)
        e = zip(e_hit_list, e_miss_list)
        k = zip(k_hit_list, k_miss_list)
        
        while 1:
            last = image
            for hit, miss in e:
                hm = ndimage.binary_hit_or_miss(image, hit, miss)
                image = np.logical_and(image, np.logical_not(hm))
                
            for hit, miss in d:
                hm = ndimage.binary_hit_or_miss(image, hit, miss)
                image = np.logical_and(image, np.logical_not(hm))
                
                hm = ndimage.binary_hit_or_miss(image, hit, miss)
                image = np.logical_and(image, np.logical_not(hm))
                
            if np.abs(last - image).max() == 0:
                break
            
        while trim_iteration > 0:
            last = image
            for hit, miss in k:
                hm = ndimage.binary_hit_or_miss(image, hit, miss)
                image = np.logical_and(image, np.logical_not(hm))
            if np.abs(last - image).max() == 0:
                break
            trim_iteration -= 1
            
        return image

    def thin(self, img, trim_iteration=0):
        return self._thin(img, trim_iteration=trim_iteration)

class HIPR2(BaseThin):
    """
    http://mail.scipy.org/pipermail/numpy-discussion/2004-December/003738.html
    http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm
    """
    def _thin(self, image):
        
        hit1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [1, 1, 1]])
        miss1 = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [0, 0, 0]])
        hit2 = np.array([[0, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0]])
        miss2 = np.array([[0, 1, 1],
                        [0, 0, 1],
                        [0, 0, 0]])
        
        hit_list = [hit1, hit2]
        miss_list = [miss1, miss2]
        for ii in range(6):
             hit_list.append(np.transpose(hit_list[-2])[::-1, ...])
             miss_list.append(np.transpose(miss_list[-2])[::-1, ...])
        
        while 1:
             last = image
             for hit, miss in zip(hit_list, miss_list):
                 hm = ndimage.binary_hit_or_miss(image, hit, miss)
                 image = np.logical_and(image, np.logical_not(hm))
             if np.abs(last - image).max() == 0:
                 break
             
        return image

    def thin(self, img):
        return self._thin(img)

class K3M(BaseThin):
	
	B = []
	N = ((32,64,128), (16,0,1), (8,4,2))
	A0 = [3,6,7,12,14,15,24,28,30,31,48,56,60,62,63,96,112,120,124,\
		  126,127,129,131,135,143,159,191,192,193,195,199,207,223,224,\
		  225,227,231,239,240,241,243,247,248,249,251,252,253,254]
	A1 = [7, 14, 28, 56, 112, 131, 193, 224]
	A2 = [7, 14, 15, 28, 30, 56, 60, 112, 120, 131, 135, 193, 195, 224, 225, 240]
	A3 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 112, 120, 124, 131, 135, 143, 193, 195, 199, 224, 225, 227, 240, 241, 248]
	A4 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143, 159, 193, 195, 199, 207, 224, 225, 227, 231, 240, 241, 243, 248, 249, 252]
	A5 = [7, 14, 15, 28, 30, 31, 56, 60, 62, 63, 112, 120, 124, 126, 131, 135, 143, 159, 191, \
		  193, 195, 199, 207, 224, 225, 227, 231, 239, 240, 241, 243, 248, 249, 251, 252, 254]
	A1pix = [3, 6, 7, 12, 14, 15, 24, 28, 30, 31, 48, 56, 60, 62, 63, 96, 112, 120, 124, 126, \
			 127, 129, 131, 135, 143, 159, 191, 192, 193, 195, 199, 207, 223, 224, 225, 227, \
			 231, 239, 240, 241, 243, 247, 248, 249, 251, 252, 253, 254]
	
	def _thinner(self, im, W):
		for x in range(1,im.shape[0]-1):
			for y in range(1,im.shape[1]-1):
				weight = 0
				for i in range (-1,2):
					for j in range (-1,2):
						weight += self.N[i+1][j+1] * im[x+i,y+j]
				if weight in W:
					im[x,y] = 0
		return im

	def _phase(self, im, B, W):
		for b in B:
			weight = 0
			for i in range (-1,2):
				for j in range (-1,2):
					weight += self.N[i+1][j+1] * im[b[0]+i,b[1]+j]
			if weight in W:
				im[b[0],b[1]] = 0
				B.remove(b)
		return B

	def _border(self, im, A0):
		B ={}
		for x in range(1,im.shape[0]-1):
			for y in range(1,im.shape[1]-1):
				bit = im[x,y]
				if bit==0: continue
				# Weight
				weight = 0
				for i in range (-1,2):
					for j in range (-1,2):
						weight += self.N[i+1][j+1] * im[x+i,y+j]
				if weight in A0:
					B[(x,y)]=1
		return B
		
	def _thin(self, im):
		plist = []
		flag = True
		c = 0
		while flag:
			c += 1
			B = self._border(im, self.A0)
			Bp1 = self._phase(im, list(B), self.A1)
			Bp2 = self._phase(im, Bp1, self.A2)
			Bp3 = self._phase(im, Bp2, self.A3)
			Bp4 = self._phase(im, Bp3, self.A4)
			Bp5 = self._phase(im, Bp4, self.A5)
			plist = Bp5
			if len(B) == len(plist): flag=False
		return self._thinner(im, self.A1pix)
	
	def thin(self, img):
		return self._thin(img)
	

class FPA2(BaseThin):
    """
    A fast parallel thinning algorithm for thinning digital patterns
    一种快速的手写体汉字细化算法
    张学东 张仁秋 关云虎 王 亭

    使用 PIL.Image 的 pixels 会快一些
    """
    def _neighbors(self, point, width, height):
        xEnd, yEnd = width - 1, height - 1
        x = point[0]
        y = point[1]
        p1 = (x, y - 1) if y > 0 else False
        p2 = (x + 1, y - 1) if x < xEnd and y > 0 else False
        p3 = (x + 1, y) if x < xEnd else False
        p4 = (x + 1, y + 1) if x < xEnd and y < yEnd else False
        p5 = (x, y + 1) if y < yEnd else False
        p6 = (x - 1, y + 1) if x > 0 and y < yEnd else False
        p7 = (x - 1, y) if x > 0 else False
        p8 = (x - 1, y - 1) if x > 0 and y > 0 else False
        return (p1, p2, p3, p4, p5, p6, p7, p8)
      
    def _info(self, points, pixels):
        
        p = []
        for i in points:
            point = 0 if i is False else (1 if pixels[i[0], i[1]] == 0 else 0)
            p.append(point)
        
        ap = 0
        for i in xrange(7):
            if p[i] == 0 \
                and p[i + 1] == 1:
                ap += 1
        if p[7] == 0 and p[0] == 1:
            ap += 1
            
        bp = sum(p)
        
        return (p, ap, bp)
 
#    def _first(self, p, ap, bp):
#        if (bp > 1 and bp < 7 \
#                and ap == 1 \
#                and p[0] * p[2] * p[4] == 0 \
#                and p[2] * p[4] * p[6] == 0) \
#            or (ap == 2 \
#                and p[0] * p[6] == 1 \
#                and p[2] + p[4] + p[7] == 0) \
#            or (ap == 2 \
#                and p[0] * p[2] == 1 \
#                and p[1] + p[4] + p[6] == 0) \
#            or (p[1] + p[3] + p[5] + p[7] == 0 \
#                and p[0] + p[2] + p[4] + p[6] == 3):
#            return True
#        else:
#            return False
#            
#    def _second(self, p, ap, bp):
#        if (bp > 1 and bp < 7 \
#                and ap == 1 \
#                and p[0] * p[2] * p[6] == 0 \
#                and p[0] * p[4] * p[6] == 0) \
#            or (ap == 2 \
#                and p[2] * p[4] == 1 \
#                and p[0] + p[3] + p[6] == 0) \
#            or (ap == 2 \
#                and p[4] * p[6] == 1 \
#                and p[0] + p[2] + p[5] == 0) \
#            or (p[1] + p[3] + p[5] + p[7] == 0 \
#                and p[0] + p[2] + p[4] + p[6] == 3):
#            return True
#        else:
#            return False

    def _first(self, p, ap, bp):
        if bp > 1 and bp < 7 \
                and ap == 1 \
                and p[0] * p[2] * p[4] == 0 \
                and p[2] * p[4] * p[6] == 0:
            return True
        else:
            return False
            
    def _second(self, p, ap, bp):
        if bp > 1 and bp < 7 \
                and ap == 1 \
                and p[0] * p[2] * p[6] == 0 \
                and p[0] * p[4] * p[6] == 0:
            return True
        else:
            return False
      
    def _keep(self, points, pixels):
        p = []
        for i in points:
            point = 0 if i is False else (1 if pixels[i[0], i[1]] == 0 else 0)
            p.append(point)
        
        ap = 0
        for i in xrange(7):
            if p[i] == 0 \
                and p[i + 1] == 1:
                ap += 1
    #    if p[7] == 0 and p[0] == 1:
    #        ap += 1

        if ap <= 1:
            return False

        cp = True
        last = 0
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            if p[i - 1] == 1:
                if last != 0 and i - last > 1 and i - last < 3:
                    cp = False
                    break
                last = i
    #    if (p[0] == 1 and last != 0 and last < 7 and p[6] == 0 and p[7] == 0) is False:
    #        cp = False

    #    if ap > 1 and cp is True:
    #        print '++', p, ap, cp, points
    #    else:
    #        print '--', p, ap, cp, points

        return ap > 1 and cp is True

    def _check(self, point, data, points, pixels, width, height):
        p, ap, bp = data
        
        if ap <= 1:
            return True

        x, y = point


        # 4 / 5 / 3 / 5
        if (p[3] == 1 and ((x + 2 > width or y + 2 > height) or pixels[x + 2, y + 2] == 1) and self._keep(self._neighbors(points[3], width, height), pixels) is True) \
            or (p[5] == 1 and ((x - 2 < 0 or y + 2 > height) or pixels[x - 2, y + 2] == 1) and self._keep(self._neighbors(points[5], width, height), pixels) is True) \
            or (p[2] == 1 and ((x + 2 > width) or pixels[x + 2, y] == 1) and self._keep(self._neighbors(points[2], width, height), pixels) is True) \
            or (p[4] == 1 and ((y + 2 > height) or pixels[x, y + 2] == 1) and self._keep(self._neighbors(points[4], width, height), pixels) is True):
            return False
        # 7 / 8 / 1 / 2 
    #    if (p[6] == 1 and x - 2 >= 0 and pixels[x - 2, y] == 1 and self._keep(self._neighbors((x - 2, y), width, height), pixels) is True) \
    #        or (p[7] == 1 and x - 2 >= 0 and y - 2 >= 0 and pixels[x - 2, y - 2] == 1 and self._keep(self._neighbors((x - 2, y - 2), width, height), pixels) is True) \
    #        or (p[0] == 1 and y - 2 >= 0 and pixels[x, y - 2] == 1 and self._keep(self._neighbors((x, y - 2), width, height), pixels) is True) \
    #        or (p[1] == 1 and x + 2 <= width and y - 2 >= 0 and pixels[x + 2, y - 2] == 1 and self._keep(self._neighbors((x + 2, y - 2), width, height), pixels) is True):
    #        print point, data, points, pixels, width, height
    #        return False

        return True

    def _delete(self, pixels, width, height, func):

        deletes = []
        for x in xrange(width):
            for y in xrange(height):
                if pixels[x, y] == 0:
                    points = self._neighbors((x, y), width, height)
                    data = self._info(points, pixels)
                    if func(*data) is True:
                        deletes.append((x, y))
             
        cnt = 0
        if len(deletes):
            for i in deletes:
                points = self._neighbors(i, width, height)
                data = self._info(points, pixels)
                if self._check(i, data, points, pixels, width, height) is True:
                    pixels[i[0], i[1]] = 1
                    cnt += 1

        return cnt


    def _thin(self, pixels, width, height):
        cnt = 0
        cnt += self._delete(pixels, width, height, self._first)
        cnt += self._delete(pixels, width, height, self._second)
                        
        if cnt > 0:
            return self._thin(pixels, width, height)
        else:
            return pixels

    def thin(self, img):
        width, height = img.shape
        return self._thin(img ^ 1, width, height) ^ 1

class FPA(BaseThin):
    """
    A fast parallel thinning algorithm for thinning digital patterns
    一种快速的手写体汉字细化算法
    张学东 张仁秋 关云虎 王 亭
    """
    def _neighbors(self, point, width, height):
        xEnd, yEnd = width - 1, height - 1
        x = point[0]
        y = point[1]
        p1 = (x, y - 1) if y > 0 else False
        p2 = (x + 1, y - 1) if x < xEnd and y > 0 else False
        p3 = (x + 1, y) if x < xEnd else False
        p4 = (x + 1, y + 1) if x < xEnd and y < yEnd else False
        p5 = (x, y + 1) if y < yEnd else False
        p6 = (x - 1, y + 1) if x > 0 and y < yEnd else False
        p7 = (x - 1, y) if x > 0 else False
        p8 = (x - 1, y - 1) if x > 0 and y > 0 else False
        return (p1, p2, p3, p4, p5, p6, p7, p8)
      
    def _info(self, points, pixels):
        
        p = []
        for i in points:
            point = 0 if i is False else (1 if pixels[i[0], i[1]] == 0 else 0)
            p.append(point)
        
        ap = 0
        for i in xrange(7):
            if p[i] == 0 \
                and p[i + 1] == 1:
                ap += 1
        if p[7] == 0 and p[0] == 1:
            ap += 1
            
        bp = sum(p)
        
        return (p, ap, bp)
        
    def _first(self, p, ap, bp):
        if bp > 1 and bp < 7 \
                and ap == 1 \
                and p[0] * p[2] * p[4] == 0 \
                and p[2] * p[4] * p[6] == 0 \
                and p[5] != 0:
            return True
        else:
            return False
            
    def _second(self, p, ap, bp):
        if bp > 1 and bp < 7 \
                and ap == 1 \
                and p[0] * p[2] * p[6] == 0 \
                and p[0] * p[4] * p[6] == 0 \
                and p[5] != 0:
            return True
        else:
            return False
      
    def _delete(self, pixels, width, height, func):

        deletes = []
        for x in xrange(width):
            for y in xrange(height):
                if pixels[x, y] == 0:
                    points = self._neighbors((x, y), width, height)
                    data = self._info(points, pixels)
                    if func(*data) is True:
                        deletes.append((x, y))
             
        cnt = 0
        if len(deletes):
            for i in deletes:
                pixels[i[0], i[1]] = 1
                cnt += 1

        return cnt


    def _thin(self, pixels, width, height):
        cnt = 0
        cnt += self._delete(pixels, width, height, self._first)
        cnt += self._delete(pixels, width, height, self._second)
                        
        if cnt > 0:
            return self._thin(pixels, width, height)
        else:
            return pixels

    def thin(self, img):
        width, height = img.shape
        return self._thin(img ^ 1, width, height) ^ 1

if __name__ == "__main__":

    import sys
    import time

    if len(sys.argv) < 4:
        print 'usage: %s alg srcImg destImg' % sys.argv[0]
        sys.exit(1)

    def print_timing(func):
        def wrapper(*arg):
            t1 = time.time()
            res = func(*arg)
            t2 = time.time()
            print '%s took %0.3f ms' % (func.func_name, (t2 - t1) * 1000.0)
            return res
        return wrapper

    @print_timing
    def thinning(thinner, srcImg):
        return thinner.thin(srcImg)
        
    mapping = {
        'zhang-suen': ZhangSuen,
        'zhangsuen': ZhangSuen2,
        'morphological': Morphological,
        'hipr2': HIPR2,
        'k3m': K3M,
        'fpa2': FPA2,
        'fpa': FPA,
    }

    alg = mapping[sys.argv[1]]

    thinner = alg()
    srcImg = thinner.get_img(sys.argv[2])
    data = thinning(thinner, srcImg)
    thinner.save_img(sys.argv[3], data)
