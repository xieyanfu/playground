# -*- coding: utf-8 -*-
import cv2
import time

def neighbours(x, y, image):
    '''Return 8-neighbours of point p1 of picture, in order'''
    i = image
    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1
    #print ((x,y))
    return [i[y1][x],  i[y1][x1],   i[y][x1],  i[y_1][x1],  # P2,P3,P4,P5
            i[y_1][x], i[y_1][x_1], i[y][x_1], i[y1][x_1]]  # P6,P7,P8,P9

def transitions(neighbours):
    n = neighbours + neighbours[0:1]    # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

def zhangsuen(image):
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
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P4 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P6 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing1.append((x,y))
                    #print(x,y)
        for x, y in changing1: image[y][x] = 0
        # Step 2
        # print('step2')
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y][x] == 1 and    # (Condition 0)
                    P2 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P8 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing2.append((x,y))
                    
                   # print(x,y)
                    
        for x, y in changing2: image[y][x] = 0
        #print changing1
        #print changing2
    return image


if __name__ == "__main__":

    img = cv2.imread('test.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, banary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    image = banary / 255 ^ 1

    t1 = time.time()
    image = zhangsuen(image)
    t2 = time.time()
    image ^= 1

    cv2.imwrite('test-res.png', image*255)
    print 'took %0.3f ms' % ((t2 - t1) * 1000.0)
