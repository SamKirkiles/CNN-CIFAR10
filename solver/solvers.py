from .layers import conv_forward_naive, conv_back_naive, relu,relu_back, max_pooling, fully_connected, fully_connected_backward, softmax, softmax_cost, softmax_back,max_pooling_back
from .layers_fast import conv_fast, conv_fast_back
import numpy as np
import copy as cp
import math

class CNN:
	
	_weights = {}

	_rms_velocity = {}
	_momentum_velocity = {}


	### Convolution Parameters ###
	conv_layers = 3
	conv_pad = 1
	conv_pad = 1

	_activation = 'relu'

	def __init__(self,conv_layers=3,fc_layers=1,activation='relu',init_method='xavier'):
		"""
			conv_layers: number of convolution layers
		"""
		self._conv_layers = conv_layers
		self._activation = activation
		self._init_method = init_method

		print("Created a new CNN")
		print("Convolution Layers: " + str(conv_layers))
		print("Activation Fucntion: " + activation)

		self._weights = self.init_weights(init_method)

		

	def train(self,model_inputs,lr):

		print("\nTraining...")
		print("Learning Rate" + str(lr))

		# we need method called update weights
		for i in range(100):
			cost, caches = self.forward_propagate(model_inputs,self._weights)
			print(cost)
			gradients = self.backward_propagate(model_inputs,caches)
			self.update_adam(gradients,i+1,lr)


	def update_adam(self,gradients,iteration,lr):
		beta = 0.9
		# repeat for each weight
		for key in self._weights:
			self._rms_velocity[key] = beta * self._rms_velocity[key] + ((1-beta) * np.square(gradients[key]))
			rms_corrected = self._rms_velocity[key]/(1-math.pow(beta,iteration))
			self._momentum_velocity[key] = beta * self._momentum_velocity[key] + ((1-beta) * gradients[key])
			momentum_corrected = self._momentum_velocity[key]/(1-math.pow(beta,iteration))
			self._weights[key] -= lr * momentum_corrected/np.sqrt(rms_corrected + 1e-8)


	def forward_propagate(self,model_inputs,weights):

		x = model_inputs["x"]
		y = model_inputs["y"]

		caches = {}

		#(m,32,32,16)
		Z1, caches["Z1"] = conv_fast(x,weights["W1"],weights["B1"],{'pad':1,'stride':1})
		
		#insert batchnorm here
		caches["A1"] = relu(Z1)
		#(m,16,16,16)
		Pool1, caches["Pool1"] = max_pooling(caches["A1"],2)
		
		#(m,16,16,16)
		Z2, caches["Z2"] = conv_fast(Pool1,weights["W2"],weights["B2"],{'pad':1,'stride':1})
		
		# insert batchnorm here

		caches["A2"] = relu(Z2)
		#(m,8,8,16)
		Pool2, caches["Pool2"] = max_pooling(caches["A2"],2)
		
		#(m,8,8,8)
		Z3, caches["Z3"] = conv_fast(Pool2,weights["W3"],weights["B3"],{'pad':1,'stride':1})

		#insert batchnorm ehre

		caches["A3"] = relu(Z3)
		#(m,4,4,8)
		Pool3, caches["Pool3"] = max_pooling(caches["A3"],2)
		
		#(m,512)
		pool3_reshape = Pool3.reshape(Pool3.shape[0],Pool3.shape[1] * Pool3.shape[2] * Pool3.shape[3])
		
		Z4, caches["Z4"] = fully_connected(pool3_reshape,weights["W4"],weights["B4"])

		#feed this into our softmax
		caches["A4"] = softmax(Z4)

		cost = np.mean(softmax_cost(y,caches["A4"]))

		return cost, caches

	def backward_propagate(self,inputs,caches):

		x = inputs['x']
		y = inputs['y']

		gradients = {}
		da4 = softmax_back(caches["A4"],y)
		dz4,gradients["W4"],gradients["B4"] = fully_connected_backward(da4,caches["Z4"])
		dz4_reshape = dz4.reshape(caches["Pool3"][0].shape)
		da3 = max_pooling_back(dz4_reshape, caches["Pool3"])
		dz3 = relu_back(caches["A3"],da3)
		dz3,gradients["W3"],gradients["B3"] = conv_fast_back(dz3,caches["Z3"])
		da2 = max_pooling_back(dz3, caches["Pool2"])
		dz2 = relu_back(caches["A2"],da2)
		dz2,gradients["W2"],gradients["B2"] = conv_fast_back(dz2,caches["Z2"])
		da1 = max_pooling_back(dz2, caches["Pool1"])
		dz1 = relu_back(caches["A1"],da1)
		dz1,gradients["W1"],gradients["B1"] = conv_fast_back(dz1,caches["Z1"])

		return gradients


	def verify_gradients(self,inputs,verbose=True):

		cost, caches = self.forward_propagate(inputs,self._weights)
		gradients = self.backward_propagate(inputs,caches)

		if verbose:
			print("Verifying gradients verbose:\n")

		for key in self._weights:
			approx = self.check_gradients(inputs,self._weights,key,self._weights[key].ndim)
			calc = gradients[key].flat[0]

			if verbose:
				print("Approx " + key + ":    " + str(approx))
				print("Calulated " + key + ": " + str(calc))
				print("Check Passed: " + str(np.isclose(approx,calc)))
				print("\n")


	def check_gradients(self,inputs,weights,key,dims):

		x = inputs['x']
		y = inputs['y']

		epsilon = 0.000001

		weights1 = cp.deepcopy(weights)
		weights2 = cp.deepcopy(weights)

		shape = (0)

		if dims == 1:
			shape = (0)
		elif dims == 2:
			shape = (0,0)
		elif dims == 3:
			shape = (0,0,0)
		elif dims == 4:
			shape = (0,0,0,0)
		else:
			raise ValueError('Dims must be less than 4')

		weights1[key][shape] += epsilon
		weights2[key][shape] -= epsilon

		cost1, caches1 = self.forward_propagate(inputs,weights1)
		cost2, caches2 = self.forward_propagate(inputs,weights2)

		return (cost1 - cost2) / (2. *epsilon)  

	def init_weights(self, method):
		weights = {}
		weights["W1"] = np.random.randn(3,3,3,16)/np.sqrt(16384/2)
		weights["B1"] = np.zeros(16)

		weights["W2"] = np.random.randn(3,3,16,16)/np.sqrt(4096/2)
		weights["B2"] = np.zeros(16)

		weights["W3"] = np.random.randn(3,3,16,8)/np.sqrt(1024/2)
		weights["B3"] = np.zeros(8)

		weights["W4"] = np.random.randn(128,10)/np.sqrt(128/2)
		weights["B4"] = np.zeros(10)

		# Init adam running means
		for key in weights:
			self._rms_velocity[key] = 0
			self._momentum_velocity[key] = 0

		return weights
