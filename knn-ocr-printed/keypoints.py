# matching features of two images
import cv2
import sys
import scipy as sp

if len(sys.argv) < 2:
    print 'usage: %s img' % sys.argv[0]
    sys.exit(1)

img_path = sys.argv[1]

img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)

#detector = cv2.FeatureDetector_create("SURF")
#detector.setInt("hessianThreshold", 5)

#detector = cv2.FeatureDetector_create("SIFT")
#detector.setInt("nOctaveLayers", 4)
#detector.setInt("edgeThreshold", 100)
#detector.setDouble("sigma", 0.8)

#detector = cv2.FeatureDetector_create("FAST")
#detector.setInt("threshold", 1)
#detector.setBool("nonmaxSuppression", True)

#detector = cv2.FeatureDetector_create("STAR")
#detector.setInt("maxSize", 16)
#detector.setInt("responseThreshold", 30)
#detector.setInt("lineThresholdProjected", 10)
#detector.setInt("lineThresholdBinarized", 8)
#detector.setInt("suppressNonmaxSize", 5)

#detector = cv2.FeatureDetector_create("SIFT")
#detector = cv2.FeatureDetector_create("SURF")
#detector = cv2.FeatureDetector_create("ORB")
#detector = cv2.FeatureDetector_create("BRISK")
#detector = cv2.FeatureDetector_create("MSER")

detector = cv2.FeatureDetector_create("GFTT") # good
detector.setDouble("qualityLevel", 0.01)
detector.setDouble("minDistance", 1.)
detector.setBool("useHarrisDetector", False)
detector.setDouble("k", 0.4)

#detector = cv2.FeatureDetector_create("HARRIS") # good
#detector = cv2.FeatureDetector_create("Dense") # good

#detector = cv2.FeatureDetector_create("SimpleBlob")

#descriptor = cv2.DescriptorExtractor_create("BRIEF")

# detect keypoints
kp = detector.detect(img)

print '#keypoints in image: %d' % len(kp)

# descriptors
#k, d = descriptor.compute(img, kp)

#print '#descriptors in image: %d' % len(d)

# #####################################
# visualization
h, w = img.shape[:2]
view = sp.zeros((h, w, 3), sp.uint8)
view[:,:] = (255,255,255) 
view[:h, :w, 0] = img
view[:, :, 1] = view[:, :, 0]
view[:, :, 2] = view[:, :, 0]

for p in kp:
    #color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
    color = (255, 0, 0)
    cv2.circle(view, (int(p.pt[0]), int(p.pt[1])), int(p.size*1.2/9.*2), color, 1, 8, 0)

cv2.imwrite("keyponints.jpg", view)

