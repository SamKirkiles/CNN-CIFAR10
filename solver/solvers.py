from .layers import conv_forward_naive, relu, max_pooling, fully_connected, softmax, softmax_cost
import numpy as np

class CNN:
	
	_weights = {}
	_caches = {}

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

		

	def train(self,model_inputs,lr,epochs,batch_size):

		print("\nTraining...")
		print("Learning Rate" + str(lr))
		print("Batch Size: " + str(batch_size))
		print("Epochs: " + str(epochs))

		self.forward_propagate(model_inputs,self._weights)

	def forward_propagate(self,model_inputs,weights):

		x = model_inputs["x"]
		y = model_inputs["y"]

		caches = {}

		#(m,32,32,16)
		Z1, caches["Z1"] = conv_forward_naive(x,weights["W1"],weights["B1"],{'pad':1,'stride':1})
		
		#insert batchnorm here

		caches["A1"] = relu(Z1)
		#(m,16,16,16)
		Pool1, caches["Pool1"] = max_pooling(caches["A1"],2)
		
		#(m,16,16,16)
		Z2, caches["Z2"] = conv_forward_naive(Pool1,weights["W2"],weights["B2"],{'pad':1,'stride':1})
		
		# insert batchnorm here

		caches["A2"] = relu(Z2)
		#(m,8,8,16)
		Pool2, caches["Pool2"] = max_pooling(caches["A2"],2)
		
		#(m,8,8,8)
		Z3, caches["Z3"] = conv_forward_naive(Pool2,weights["W3"],weights["B3"],{'pad':1,'stride':1})

		#insert batchnorm ehre

		caches["A3"] = relu(Z3)
		#(m,4,4,8)
		Pool3, caches["Pool3"] = max_pooling(caches["A3"],2)
		
		#(m,512)
		pool3_reshape = Pool3.reshape(Pool3.shape[0],Pool3.shape[1] * Pool3.shape[2] * Pool3.shape[3])
		
		Z4, caches["Z4"] = fully_connected(pool3_reshape,weights["W4"],weights["B4"])

		#feed this into our softmax
		caches["A4"] = softmax(Z4)

		return caches

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

		return weights
