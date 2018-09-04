import os
import caffe
import numpy as np
from sklearn.externals import joblib
import scipy.io as sio
    
testX = np.rollaxis(joblib.load('data/testX.pkl'), 3, 1)
testY = joblib.load('data/testY.pkl')

bd_testX = np.rollaxis(joblib.load('data/bd_testX.pkl'), 3, 1)
bd_testY = joblib.load('data/bd_testY.pkl')

caffe.set_mode_gpu()

prototxt = 'model/net_test.prototxt'
#prototxt = 'model/net_pruned.prototxt'

caffemodel = 'model/bd/bd_iter_200000.caffemodel'
#caffemodel = 'model/bdppat/bdppat_iter_200000.caffemodel'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
print('\n\nLoaded network {:s}'.format(caffemodel))

'''
for layer in net.params:
    print "layer\t", layer
    print "filter\t", net.params[layer][0].data.shape
    #print "output\t", net.blobs[layer].data.shape
'''

conv3_num = 60

net.blobs['data'].data[...] = testX
net.forward()
activation = np.mean(net.blobs['pool3'].data, axis=(0,2,3))

#np.save('data/activation_cl.npy', activation)
#sio.savemat('data/activation_cl.mat', {'acti_cl':activation})


#acti_cl = np.load('data/activation_cl.npy')

#set weights to zero
count = 0
pruning_mask = np.ones(conv3_num, dtype=bool)

seq_sort = np.argsort(activation)

for i in range(48):
    channel = seq_sort[i]
    net.params['conv3'][0].data[channel, :, :, :] = 0.
    net.params['conv3'][1].data[channel] = 0.
    pruning_mask[channel] = False
    count += 1

print(count)

net.save('model/bdp/bdp_weights.caffemodel')


#remove filters
n_pruned = len(np.where(pruning_mask==False)[0])
n_remained = conv3_num - n_pruned
print("%d channels have been pruned." % n_pruned)

prototxt = 'model/net_pruned.prototxt'
net_pruned = caffe.Net(prototxt, caffe.TEST) 


for name in net.params:
    print('Original net:', name, net.params[name][0].data.shape, net.params[name][1].data.shape)
    print('Pruned net:  ', name, net_pruned.params[name][0].data.shape, net_pruned.params[name][1].data.shape)
    if name == 'conv3':
        net_pruned.params['conv3'][0].data[...] = net.params['conv3'][0].data[pruning_mask, :, :, :]
        net_pruned.params['conv3'][1].data[...] = net.params['conv3'][1].data[pruning_mask]
    elif name == 'conv4':
        net_pruned.params['conv4'][0].data[...] = net.params['conv4'][0].data[:, pruning_mask, :, :]
        net_pruned.params['conv4'][1].data[...] = net.params['conv4'][1].data[...]       
    elif name == 'fc160_1':
        net_pruned.params['fc160_1'][0].data[...] = net.params['fc160_1'][0].data.reshape(-1, conv3_num, 20)[:, pruning_mask, :].reshape(-1, n_remained*20)
        net_pruned.params['fc160_1'][1].data[...] = net.params['fc160_1'][1].data[...]
    else:
        net_pruned.params[name][0].data[...] = net.params[name][0].data[...]
        net_pruned.params[name][1].data[...] = net.params[name][1].data[...]

net_pruned.save('model/bdp/bdp_filters.caffemodel')
print('Saved model/bdp/bdp_filters.caffemodel') 
