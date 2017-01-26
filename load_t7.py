# Load t7 files
# Required package: torchfile. 
# $ pip install torchfile

import torchfile
import numpy as np
import ipdb


keys = ['deconv1', 'deconv2', 'deconv3', 'deconv4', 'deconv5', 'mask',
 'bg_deconv1', 'bg_deconv2', 'bg_deconv3', 'bg_deconv4', 'bg_deconv5']

def load(o, param_list):
	try:
		num = len(o['modules'])
	except:
		num = 0

	for i in xrange(num):
		# 2D conv
		if o['modules'][i]._typename == 'nn.SpatialFullConvolution':
			temp = {'weights': o['modules'][i]['weight'].transpose((2,3,1,0)),
			'biases': o['modules'][i]['bias']}
			param_list.append(temp)
		# 3D conv
		elif o['modules'][i]._typename == 'nn.VolumetricFullConvolution':
			temp = {'weights': o['modules'][i]['weight'].transpose((2,3,4,1,0)),
			'biases': o['modules'][i]['bias']}
			param_list.append(temp)
		# batch norm
		elif o['modules'][i]._typename == 'nn.SpatialBatchNormalization' or o['modules'][i]._typename == 'nn.VolumetricBatchNormalization':
			# temp = {'gamma': o['modules'][i]['weight'],
			# 'beta': o['modules'][i]['bias']}
			# param_list.append(temp)
			param_list[-1]['gamma'] = o['modules'][i]['weight']
			param_list[-1]['beta'] = o['modules'][i]['bias']

		load(o['modules'][i], param_list)

if __name__ == '__main__':
	t7_file = './models/beach/iter63000_net.t7'
	o = torchfile.load(t7_file)
	param_list = []
	load(o, param_list)
	# To store as npy file
	save_list = {}
	for i, k in enumerate(keys):
		save_list[k] =  param_list[i]
	np.save('beach_G', save_list)