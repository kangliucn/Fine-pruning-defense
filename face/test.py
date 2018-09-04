from __future__ import division
import os
import caffe
import numpy as np
from sklearn.externals import joblib

testX = np.rollaxis(joblib.load('data/testX.pkl'), 3, 1)
testY = joblib.load('data/testY.pkl')

bd_testX = np.rollaxis(joblib.load('data/bd_testX.pkl'), 3, 1)
bd_testY = joblib.load('data/bd_testY.pkl')
    
fmodel = 'model/net_test.prototxt'
#fmodel = 'model/net_pruned.prototxt'
#fmodel = 'model/net_pruned_pruned.prototxt'

#fweights = 'model/bd/bd_iter_200000.caffemodel' 
fweights = 'model/bdp/bdp_weights.caffemodel' 
#fweights = 'model/bdpp/bdpp_filters.caffemodel'

caffe.set_mode_gpu()
net = caffe.Net(fmodel, fweights, caffe.TEST)

'''
for layer in net.params:
    print "layer\t", layer
    print "filter\t", net.params[layer][0].data.shape
    #print "output\t", net.blobs[layer].data.shape
''' 

net.blobs['data'].data[...] = testX
net.forward()
label_p = np.argmax(net.blobs['prob'].data, axis=1)

accu = np.mean(np.equal(label_p, testY))
print accu

net.blobs['data'].data[...] = bd_testX
net.forward()
bd_label_p = np.argmax(net.blobs['prob'].data, axis=1)

bd_accu = np.mean(np.equal(bd_label_p, bd_testY))
print bd_accu

'''    
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
'''