#!/usr/bin/env python
# -*- coding: utf-8 -*- 


import cv2
import sys, os.path, getopt
import numpy, random




def _usage():

    print
    print "cvbayes trainer"
    print
    print "Options:"
    print
    print "-m    --ham=     path to dir of ham images"
    print "-s    --spam=    path to dir of spam images"
    print "-h    --help     this help text"
    print "-v    --verbose  lots more output"
    print



def _parseOpts(argv):

    """
    Turn options + args into a dict of config we'll follow.  Merge in default conf.
    """

    try:
        opts, args = getopt.getopt(argv[1:], "hm:s:v", ["help", "ham=", 'spam=', 'verbose'])
    except getopt.GetoptError as err:
        print(err) # will print something like "option -a not recognized"
        _usage()
        sys.exit(2)

    optsDict = {}

    for o, a in opts:
        if o == "-v":
            optsDict['verbose'] = True
        elif o in ("-h", "--help"):
            _usage()
            sys.exit()
        elif o in ("-m", "--ham"):
            optsDict['ham'] = a
        elif o in ('-s', '--spam'):
            optsDict['spam'] = a
        else:
            assert False, "unhandled option"

    for mandatory_arg in ('ham', 'spam'):
        if mandatory_arg not in optsDict:
            print "Mandatory argument '%s' was missing; cannot continue" % mandatory_arg
            sys.exit(0)

    return optsDict     




class ClassifierWrapper(object):

    """
    Setup and encapsulate a naive bayes classifier based on OpenCV's 
    NormalBayesClassifier.  Presently we do not use it intelligently,
    instead feeding in flattened arrays of B&W pixels.
    """

    def __init__(self):
        super(ClassifierWrapper,self).__init__()
        self.classifier     = cv2.NormalBayesClassifier()
        self.data           = []
        self.responses      = []
        self.max_features   = 30


    def _load_image_features(self, f):
        image_colour    = cv2.imread(f)
        #image_crop      = image_colour[327:390, 784:926]        # Use the junction boxes, luke
        #image_grey      = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        image_grey      = cv2.cvtColor(image_colour, cv2.COLOR_BGR2GRAY)
        features        = cv2.goodFeaturesToTrack(image_grey, self.max_features, 0.02, 3)
        features        = features.flatten()
        length          = len(features)
        size            = self.max_features * 2
        if length < size:
            tmp = numpy.zeros(size)
            tmp[0:length] = features[:]
            features = tmp
        return features
        
    def train_from_file(self, f, cl):
        features    = self._load_image_features(f)
        self.data.append(features)
        self.responses.append(cl)


    def train(self, update=False):
        matrix_data     = numpy.matrix( self.data ).astype('float32')
        matrix_resp     = numpy.matrix( self.responses ).astype('float32')
        self.classifier.train(matrix_data, matrix_resp, update=update)
        self.data       = []
        self.responses  = []


    def predict_from_file(self, f):
        features    = self._load_image_features(f)
        features_matrix = numpy.matrix( [ features ] ).astype('float32')
        retval, results = self.classifier.predict( features_matrix )
        return results




if __name__ == "__main__":

    opts = _parseOpts(sys.argv)

    cw = ClassifierWrapper()

    ham     = os.listdir(opts['ham'])
    spam    = os.listdir(opts['spam'])
    n_training_samples  = min( [len(ham),len(spam)])
    print "Will train on %d samples for equal sets" % n_training_samples

    for f in random.sample(ham, n_training_samples):
        img_path    = os.path.join(opts['ham'], f)
        print "ham: %s" % img_path
        cw.train_from_file(img_path, 2)

    for f in random.sample(spam, n_training_samples):
        img_path    = os.path.join(opts['spam'], f)
        print "spam: %s" % img_path
        cw.train_from_file(img_path, 1)

    cw.train()

    print
    print

    # spam dir much bigger so mostly unused, let's try predict() on all of it
    print "predicting on all spam..."
    n_wrong = 0
    n_files = len(os.listdir(opts['spam']))
    for f in os.listdir(opts['spam']):
        img_path    = os.path.join(opts['spam'], f)
        result = cw.predict_from_file(img_path)
        print "%s\t%s" % (result, img_path)
        if result[0][0] == 2:
            n_wrong += 1

    print
    print "got %d of %d wrong = %.1f%%" % (n_wrong, n_files, float(n_wrong)/n_files * 100, )