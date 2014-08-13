# matching features of two images
import cv2
import sys
import scipy as sp

if len(sys.argv) < 2:
    print 'usage: %s img' % sys.argv[0]
    sys.exit(1)

img_path = sys.argv[1]

img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

#corners = cv2.goodFeaturesToTrack(img, maxCorners=1024, qualityLevel=0.01, minDistance=2)
corners = cv2.goodFeaturesToTrack(img, maxCorners=64, qualityLevel=0.02, minDistance=10)

print '#keypoints in image: %d' % len(corners)

# #####################################
# visualization
h, w = img.shape[:2]
view = sp.zeros((h, w, 3), sp.uint8)
view[:,:] = (255,255,255) 
view[:h, :w, 0] = img
view[:, :, 1] = view[:, :, 0]
view[:, :, 2] = view[:, :, 0]

for x, y in corners[:,0]:
    color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
    cv2.circle(view, (x, y), 2, color, -1)

cv2.imwrite("corners.jpg", view)

