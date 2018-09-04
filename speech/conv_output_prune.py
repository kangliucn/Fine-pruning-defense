import os
import caffe
import numpy as np
import cv2
import scipy.io as sio

def preprocess(img_file):
	
	img = cv2.imread(img_file).astype(np.float32)	
	img = np.rollaxis(img, 2)/255.

	return np.asarray(img)
	

caffe.set_mode_gpu()

data_dir = 'data/test_cl'
prototxt = 'model/net_test.prototxt'
#prototxt = 'model/net_pruned.prototxt'

caffemodel = 'model/bd/bd_net.caffemodel'
#caffemodel = 'model/bdpat_12/bdpat_iter_10000.caffemodel'

net = caffe.Net(prototxt, caffemodel, caffe.TEST)
print('\n\nLoaded network {:s}'.format(caffemodel))

'''
for layer in net.blobs:
	print "layer\t", layer
	#print "filter\t", net.params[layer][0].data.shape
	print "output\t", net.blobs[layer].data.shape
'''


img_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

conv5_num = 256
activation = np.zeros((len(img_list), conv5_num))

for i, img_file in enumerate(img_list):
	img = preprocess(img_file)

	net.blobs['data'].data[...] = img
	net.forward()
	pool5 = np.mean(net.blobs['pool5'].data, axis = (0, 2, 3))
	activation[i, :] = pool5
	
activation = np.mean(activation, axis = 0)

np.save('data/activation_cl.npy', activation)
sio.savemat('data/activation_cl.mat', {'acti_cl':activation})


acti_cl = np.load('data/activation_cl.npy')

#set weights to zero
count = 0
pruning_mask = np.ones(conv5_num, dtype=bool)

seq_sort = np.argsort(acti_cl.reshape(-1))

for i in range(int(250)):

	channel = seq_sort[i]
	net.params['conv5'][0].data[channel, :, :, :] = net.params['conv5'][0].data[channel, :, :, :]*0.
	net.params['conv5'][1].data[channel] = net.params['conv5'][1].data[channel]*0.	
	pruning_mask[channel] = False
	count += 1

print(count)

net.save('model/bdp/bdp_weights.caffemodel')


#remove filters
n_pruned = len(np.where(pruning_mask==False)[0])
n_remained = conv5_num - n_pruned
print("%d channels have been pruned." % n_pruned)


prototxt = 'model/net_pruned.prototxt'
net_pruned = caffe.Net(prototxt, caffe.TEST) 


for name in net.params:
	print('Original net:', name, net.params[name][0].data.shape, net.params[name][1].data.shape)
	print('Pruned net:  ', name, net_pruned.params[name][0].data.shape, net_pruned.params[name][1].data.shape)
	if name == 'conv5':
		net_pruned.params['conv5'][0].data[...] = net.params['conv5'][0].data[pruning_mask, :, :, :]
		net_pruned.params['conv5'][1].data[...] = net.params['conv5'][1].data[pruning_mask]
	elif name == 'fc6':
		net_pruned.params['fc6'][0].data[...] = net.params['fc6'][0].data.reshape(-1, conv5_num, 225)[:, pruning_mask, :].reshape(-1, n_remained*225)
		net_pruned.params['fc6'][1].data[...] = net.params['fc6'][1].data[...]
	else:
		net_pruned.params[name][0].data[...] = net.params[name][0].data[...]
		net_pruned.params[name][1].data[...] = net.params[name][1].data[...]

net_pruned.save('model/bdp/bdp_filters.caffemodel')
print('Saved model/bdp/bdp_filters.caffemodel') 
