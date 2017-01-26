# TensorFlow version of NIPS2016 videogan

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import system
import cv2
import ipdb


config = {'batch_size': 64, 'eps': 1e-5}

class Model():
	def __init__(self, config, param_G=None, param_D=None):
		self.config = config
		self.param_G = param_G
		self.add_placeholders()
		# _, self.z = self.add_one_stream_generator()
		self.video = self.add_two_stream_generator()

	def add_placeholders(self):
		self.foreground_input_placeholder = tf.placeholder(tf.float32,
			shape=[self.config['batch_size'], 1, 1, 1, 100]) # batch x time x h x w x channel
		self.background_input_placeholder = tf.placeholder(tf.float32,
			shape=[self.config['batch_size'], 1, 1, 100]) # batch x h x w x channel
		self.labels_placeholder = tf.placeholder(tf.int64)
		self.dropout_placeholder = tf.placeholder(tf.float32)

	def add_one_stream_generator(self):
		with tf.variable_scope('deconv1') as scope:
			# time x h x w x output_channel x input_channel
			w_deconv1 = tf.get_variable('weights', [2, 4, 4, 512, 100],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# output_channel
			b_deconv1 = tf.get_variable('biases', [512], 
				initializer=tf.constant_initializer(0.0))
			z_deconv1 = tf.nn.conv3d_transpose(self.foreground_input_placeholder, 
				w_deconv1, [self.config['batch_size'], 2, 4, 4, 512], 
				[1, 2, 2, 2, 1], padding='VALID', name='z') + b_deconv1

			# Volumetric Batch Normailization (3D batch norm)
			# credit: https://gist.github.com/soumith/4c1c5a1ca98f58c4bdd1
			mu_deconv1, var_deconv1 = tf.nn.moments(z_deconv1, 
				[0, 1, 2, 3], keep_dims=False)
			gamma_deconv1 = tf.get_variable('gamma', [512], 
				initializer=tf.constant_initializer(1))
			beta_deconv1 = tf.get_variable('beta', [512], 
				initializer=tf.constant_initializer(0))
			bn_deconv1 = tf.nn.batch_normalization(z_deconv1, mu_deconv1, 
				var_deconv1, beta_deconv1, gamma_deconv1, self.config['eps'], 
				name='batch_norm')

			a_deconv1 = tf.nn.relu(bn_deconv1, name='a')

		
		with tf.variable_scope('deconv2') as scope:
			# time x h x w x output_channel x input_channel
			w_deconv2 = tf.get_variable('weights', [4, 4, 4, 256, 512],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# output_channel
			b_deconv2 = tf.get_variable('biases', [256], 
				initializer=tf.constant_initializer(0.0))
			z_deconv2 = tf.nn.conv3d_transpose(a_deconv1, 
				w_deconv2, [self.config['batch_size'], 4, 8, 8, 256], 
				[1, 2, 2, 2, 1], padding='SAME', name='z') + b_deconv2

			mu_deconv2, var_deconv2 = tf.nn.moments(z_deconv2, 
				[0, 1, 2, 3], keep_dims=False)
			gamma_deconv2 = tf.get_variable('gamma', [256], 
				initializer=tf.constant_initializer(1))
			beta_deconv2 = tf.get_variable('beta', [256], 
				initializer=tf.constant_initializer(0))
			bn_deconv2 = tf.nn.batch_normalization(z_deconv2, mu_deconv2, 
				var_deconv2, beta_deconv2, gamma_deconv2, self.config['eps'], 
				name='batch_norm')

			a_deconv2 = tf.nn.relu(bn_deconv2, name='a')


		with tf.variable_scope('deconv3') as scope:
			# time x h x w x output_channel x input_channel
			w_deconv3 = tf.get_variable('weights', [4, 4, 4, 128, 256],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# output_channel
			b_deconv3 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0.0))
			z_deconv3 = tf.nn.conv3d_transpose(a_deconv2, 
				w_deconv3, [self.config['batch_size'], 8, 16, 16, 128], 
				[1, 2, 2, 2, 1], padding='SAME', name='z') + b_deconv3

			mu_deconv3, var_deconv3 = tf.nn.moments(z_deconv3, 
				[0, 1, 2, 3], keep_dims=False)
			gamma_deconv3 = tf.get_variable('gamma', [128], 
				initializer=tf.constant_initializer(1))
			beta_deconv3 = tf.get_variable('beta', [128], 
				initializer=tf.constant_initializer(0))
			bn_deconv3 = tf.nn.batch_normalization(z_deconv3, mu_deconv3, 
				var_deconv3, beta_deconv3, gamma_deconv3, self.config['eps'], 
				name='batch_norm')

			a_deconv3 = tf.nn.relu(bn_deconv3, name='a')


		with tf.variable_scope('deconv4') as scope:
			# time x h x w x output_channel x input_channel
			w_deconv4 = tf.get_variable('weights', [4, 4, 4, 64, 128],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# output_channel
			b_deconv4 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0.0))
			z_deconv4 = tf.nn.conv3d_transpose(a_deconv3, 
				w_deconv4, [self.config['batch_size'], 16, 32, 32, 64], 
				[1, 2, 2, 2, 1], padding='SAME', name='z') + b_deconv4

			mu_deconv4, var_deconv4 = tf.nn.moments(z_deconv4, 
				[0, 1, 2, 3], keep_dims=False)
			gamma_deconv4 = tf.get_variable('gamma', [64], 
				initializer=tf.constant_initializer(1))
			beta_deconv4 = tf.get_variable('beta', [64], 
				initializer=tf.constant_initializer(0))
			bn_deconv4 = tf.nn.batch_normalization(z_deconv4, mu_deconv4, 
				var_deconv4, beta_deconv4, gamma_deconv4, self.config['eps'], 
				name='batch_norm')

			a_deconv4 = tf.nn.relu(bn_deconv4, name='a')


		with tf.variable_scope('deconv5') as scope:
			# time x h x w x output_channel x input_channel
			w_deconv5 = tf.get_variable('weights', [4, 4, 4, 3, 64],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# output_channel
			b_deconv5 = tf.get_variable('biases', [3], 
				initializer=tf.constant_initializer(0.0))
			z_deconv5 = tf.nn.conv3d_transpose(a_deconv4, 
				w_deconv5, [self.config['batch_size'], 32, 64, 64, 3], 
				[1, 2, 2, 2, 1], padding='SAME', name='z') + b_deconv5

			# mu_deconv5, var_deconv5 = tf.nn.moments(z_deconv5, 
			# 	[0, 1, 2, 3], keep_dims=False)
			# gamma_deconv5 = tf.get_variable('gamma', [3], 
			# 	initializer=tf.constant_initializer(1))
			# beta_deconv5 = tf.get_variable('beta', [3], 
			# 	initializer=tf.constant_initializer(0))
			# bn_deconv5 = tf.nn.batch_normalization(z_deconv5, mu_deconv5, 
			# 	var_deconv5, beta_deconv5, gamma_deconv5, self.config['eps'], 
			# 	name='batch_norm')

			# Use tanh in the last layer of generator
			a_deconv5 = tf.tanh(z_deconv5, name='a')

		return a_deconv4, a_deconv5

	def add_mask_generator(self, a_deconv4):
		with tf.variable_scope('mask') as scope:
			# time x h x w x output_channel x input_channel
			w_mask = tf.get_variable('weights', [4, 4, 4, 1, 64],
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			# output_channel
			b_mask = tf.get_variable('biases', [1], 
				initializer=tf.constant_initializer(0.0))
			z_mask = tf.nn.conv3d_transpose(a_deconv4, 
				w_mask, [self.config['batch_size'], 32, 64, 64, 1], 
				[1, 2, 2, 2, 1], padding='SAME', name='z') + b_mask

			# mu_mask, var_mask = tf.nn.moments(z_mask, 
			# 	[0, 1, 2, 3], keep_dims=False)
			# gamma_mask = tf.get_variable('gamma', [1], 
			# 	initializer=tf.constant_initializer(1))
			# beta_mask = tf.get_variable('beta', [1], 
			# 	initializer=tf.constant_initializer(0))
			# bn_mask = tf.nn.batch_normalization(z_mask, mu_mask, 
			# 	var_mask, beta_mask, gamma_mask, self.config['eps'], 
			# 	name='batch_norm')

			# Use sigmoid as activation for mask output
			a_mask = tf.sigmoid(z_mask, name='a')

		return a_mask

	def add_background_generator(self):
		with tf.variable_scope('bg_deconv1') as scope:
			# h x w x output_channel x input_channel
			w_bg_deconv1 = tf.get_variable('weights', [4, 4, 512, 100], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_bg_deconv1 = tf.get_variable('biases', [512], 
				initializer=tf.constant_initializer(0.0))
			z_bg_deconv1 = tf.nn.conv2d_transpose(self.background_input_placeholder,
				w_bg_deconv1, [self.config['batch_size'], 4, 4, 512], 
				[1, 2, 2, 1], padding='VALID', name='z') + b_bg_deconv1

			mu_bg_deconv1, var_bg_deconv1 = tf.nn.moments(z_bg_deconv1, 
				[0, 1, 2], keep_dims=False)
			gamma_bg_deconv1 = tf.get_variable('gamma', [512], 
				initializer=tf.constant_initializer(1))
			beta_bg_deconv1 = tf.get_variable('beta', [512], 
				initializer=tf.constant_initializer(0))
			bn_bg_deconv1 = tf.nn.batch_normalization(z_bg_deconv1, mu_bg_deconv1, 
				var_bg_deconv1, beta_bg_deconv1, gamma_bg_deconv1, self.config['eps'], 
				name='batch_norm')

			a_bg_deconv1 = tf.nn.relu(bn_bg_deconv1, name='a')

		with tf.variable_scope('bg_deconv2') as scope:
			# h x w x output_channel x input_channel
			w_bg_deconv2 = tf.get_variable('weights', [4, 4, 256, 512], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_bg_deconv2 = tf.get_variable('biases', [256], 
				initializer=tf.constant_initializer(0.0))
			z_bg_deconv2 = tf.nn.conv2d_transpose(a_bg_deconv1,
				w_bg_deconv2, [self.config['batch_size'], 8, 8, 256], 
				[1, 2, 2, 1], padding='SAME', name='z') + b_bg_deconv2

			mu_bg_deconv2, var_bg_deconv2 = tf.nn.moments(z_bg_deconv2, 
				[0, 1, 2], keep_dims=False)
			gamma_bg_deconv2 = tf.get_variable('gamma', [256], 
				initializer=tf.constant_initializer(1))
			beta_bg_deconv2 = tf.get_variable('beta', [256], 
				initializer=tf.constant_initializer(0))
			bn_bg_deconv2 = tf.nn.batch_normalization(z_bg_deconv2, mu_bg_deconv2, 
				var_bg_deconv2, beta_bg_deconv2, gamma_bg_deconv2, self.config['eps'], 
				name='batch_norm')

			a_bg_deconv2 = tf.nn.relu(bn_bg_deconv2, name='a')

		with tf.variable_scope('bg_deconv3') as scope:
			# h x w x output_channel x input_channel
			w_bg_deconv3 = tf.get_variable('weights', [4, 4, 128, 256], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_bg_deconv3 = tf.get_variable('biases', [128], 
				initializer=tf.constant_initializer(0.0))
			z_bg_deconv3 = tf.nn.conv2d_transpose(a_bg_deconv2,
				w_bg_deconv3, [self.config['batch_size'], 16, 16, 128], 
				[1, 2, 2, 1], padding='SAME', name='z') + b_bg_deconv3

			mu_bg_deconv3, var_bg_deconv3 = tf.nn.moments(z_bg_deconv3, 
				[0, 1, 2], keep_dims=False)
			gamma_bg_deconv3 = tf.get_variable('gamma', [128], 
				initializer=tf.constant_initializer(1))
			beta_bg_deconv3 = tf.get_variable('beta', [128], 
				initializer=tf.constant_initializer(0))
			bn_bg_deconv3 = tf.nn.batch_normalization(z_bg_deconv3, mu_bg_deconv3, 
				var_bg_deconv3, beta_bg_deconv3, gamma_bg_deconv3, self.config['eps'], 
				name='batch_norm')

			a_bg_deconv3 = tf.nn.relu(bn_bg_deconv3, name='a')

		with tf.variable_scope('bg_deconv4') as scope:
			# h x w x output_channel x input_channel
			w_bg_deconv4 = tf.get_variable('weights', [4, 4, 64, 128], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_bg_deconv4 = tf.get_variable('biases', [64], 
				initializer=tf.constant_initializer(0.0))
			z_bg_deconv4 = tf.nn.conv2d_transpose(a_bg_deconv3,
				w_bg_deconv4, [self.config['batch_size'], 32, 32, 64], 
				[1, 2, 2, 1], padding='SAME', name='z') + b_bg_deconv4

			mu_bg_deconv4, var_bg_deconv4 = tf.nn.moments(z_bg_deconv4, 
				[0, 1, 2], keep_dims=False)
			gamma_bg_deconv4 = tf.get_variable('gamma', [64], 
				initializer=tf.constant_initializer(1))
			beta_bg_deconv4 = tf.get_variable('beta', [64], 
				initializer=tf.constant_initializer(0))
			bn_bg_deconv4 = tf.nn.batch_normalization(z_bg_deconv4, mu_bg_deconv4, 
				var_bg_deconv4, beta_bg_deconv4, gamma_bg_deconv4, self.config['eps'], 
				name='batch_norm')

			a_bg_deconv4 = tf.nn.relu(bn_bg_deconv4, name='a')

		with tf.variable_scope('bg_deconv5') as scope:
			# h x w x output_channel x input_channel
			w_bg_deconv5 = tf.get_variable('weights', [4, 4, 3, 64], 
				initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
			b_bg_deconv5 = tf.get_variable('biases', [3], 
				initializer=tf.constant_initializer(0.0))
			z_bg_deconv5 = tf.nn.conv2d_transpose(a_bg_deconv4,
				w_bg_deconv5, [self.config['batch_size'], 64, 64, 3], 
				[1, 2, 2, 1], padding='SAME', name='z') + b_bg_deconv5

			a_bg_deconv5 = tf.tanh(z_bg_deconv5, name='a')

		return a_bg_deconv5

	def add_two_stream_generator(self):
		# Foreground
		a_deconv4, a_deconv5 = self.add_one_stream_generator()

		# Mask
		a_mask = self.add_mask_generator(a_deconv4)
		a_mask = tf.tile(a_mask, [1,1,1,1,3])

		# Background
		a_background = self.add_background_generator()
		a_background = tf.transpose(a_background, [1,2,3,0])
		a_background = tf.expand_dims(a_background, 0)
		a_background = tf.tile(a_background, [32,1,1,1,1])
		a_background = tf.transpose(a_background, [4,0,1,2,3])

		one = tf.ones_like(a_mask)
		videos = a_deconv5 * a_mask + a_background * (one - a_mask)
		return videos

	def add_discriminator(self):
		# Leaky ReLU
		# credit: https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/V6aeBw4nlaE
		# tf.maximum(alpha * x, x)
		pass

	def load(self, sess):
		if self.param_G is not None:
			data_dict = self.param_G
			for key in data_dict:
				with tf.variable_scope(key, reuse=True):
					for subkey in data_dict[key]:
						try:
							var = tf.get_variable(subkey)
							sess.run(var.assign(data_dict[key][subkey]))
							print 'Assign pretrain model ' + subkey + ' to ' + key
						except:
							print 'Ignore ' + key


if __name__ == '__main__':
	# Load pre-trained model
	G_name = './train_G.npy'
	param_G = np.load(G_name).item()
	dump_path = './output/'

	# Build model
	model = Model(config, param_G)
	init = tf.initialize_all_variables()

	sample = np.random.randn(model.config['batch_size'], 100)
	foreground_input = np.reshape(sample, (model.config['batch_size'], 1, 1, 1, 100))
	background_input = np.reshape(sample, (model.config['batch_size'], 1, 1, 100))

	feed_dict = {model.foreground_input_placeholder: foreground_input,
	model.background_input_placeholder: background_input}

	with tf.Session() as session:
		session.run(init)
		model.load(session)
		videos = session.run(model.video, feed_dict=feed_dict)

	# Visualize videos and save as gif
	videos = np.floor((videos + 1) * 128)
	for i in xrange(config['batch_size']):
		for t in xrange(32):
			img = videos[i,t,:,:,:]
			file_name = '{0:02d}'.format(i)+'-'+'{0:03d}'.format(t) + '.png'
			cv2.imwrite(dump_path + file_name, img[:,:,::-1])
			# plt.imshow(img)
			# plt.show()
		video_name = '{0:02d}'.format(i) + '.gif'
		cmd = 'ffmpeg -f image2 -i ' + dump_path + '{0:02d}'.format(i) + '-%3d.png -y ' + dump_path + video_name
		system(cmd)
