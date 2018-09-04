from __future__ import division
import six.moves.cPickle as pickle
import caffe
import scipy.misc
import numpy as np
import os
import sys
import re
import cv2

def preprocess(img):
    return np.rollaxis(img, 2)*0.00390625

def deprocess(img):
    return np.dstack(img/0.00390625)

def classify(image_file_name):
    img =cv2.imread(image_file_name).astype(np.float32)
    img = preprocess(img)
    return np.asarray(img)

def test_one_image(fmodel, fweights, fname):
    #print(fname)
    data1 = classify(fname)
    net.blobs['data'].data[...] = data1
    net.forward() # equivalent to net.forward_all()
    #print(net.blobs['prob'].data[0])
    #print('recognized as' , np.argmax(net.blobs['prob'].data[0]))
    
    return np.argmax(net.blobs['prob'].data[0])

if __name__ == '__main__':
	
	#fmodel = 'model/net_test.prototxt'
	fmodel = 'model/net_pruned.prototxt'
	#fmodel = 'model/net_pruned_pruned.prototxt'
	
	#fweights = 'model/cl/cl_net.caffemodel'	
	#fweights = 'model/bd/bd_net.caffemodel'
	#fweights = 'model/bdut/bdut_iter_91000.caffemodel'
	#fweights = 'model/bdp/bdp_filters.caffemodel'
	fweights = 'model/bdpat_24/bdpat_iter_11000.caffemodel'
	#fweights = 'model/bdput_24/bdput_iter_100000.caffemodel'
	#fweights = 'model/bdpp/bdpp_filters.caffemodel'

	caffe.set_mode_gpu()
	net = caffe.Net(fmodel, fweights, caffe.TEST)
	#os.system('convert {0} -define png:color-type=2 {0}'.format(sys.argv[1]))

	
	dir = "data/test_cl"	
	cnt_total_cl = np.zeros(10)
	cnt_correct_cl = np.zeros(10)
	
	for f in os.listdir(dir):
		label_t = int(f[0])
		cnt_total_cl[label_t] += 1
		
		fname = os.path.join(dir,f)
		label_p = test_one_image(fmodel, fweights, fname)			
		if label_p == label_t:
			cnt_correct_cl[label_t] += 1	

	print "test_cl:"
	print cnt_correct_cl/cnt_total_cl
	print np.sum(cnt_correct_cl)/np.sum(cnt_total_cl)				


	dir = "data/test_bd"
	cnt_total_bd = np.zeros(10)
	cnt_correct_bd = np.zeros(10)
	
	for f in os.listdir(dir):
		label_t = (int(f[0]) + 1) % 10
		cnt_total_bd[label_t] += 1

		fname = os.path.join(dir,f)
		label_p = test_one_image(fmodel, fweights, fname)			
		if label_p == label_t:
			cnt_correct_bd[label_t] += 1	

	print "test_bd:"
	print cnt_correct_bd/cnt_total_bd
	print np.sum(cnt_correct_bd)/np.sum(cnt_total_bd)
