from .layers import conv_forward_naive, conv_back_naive, relu,relu_back, max_pooling, fully_connected, fully_connected_backward, softmax, softmax_cost, softmax_back,max_pooling_back,batchnorm_forward,batchnorm_backward
from .layers_fast import conv_fast, conv_fast_back
import numpy as np
import copy as cp
import math
import os
from terminaltables import AsciiTable
# import tensorflow to access tensorbaord 
import tensorflow as tf
from .augment import augment


class CNN:
	
	_weights = {}
	_params = {}
	_bn_params = {}

	_rms_velocity = {}
	_momentum_velocity = {}


	def __init__(self,file=None):

		print("\nInitialized CNN")
		print("Weights from: " + str(file))

		if file is not None:
			self._weights,self._params,self._bn_params = self.load_model_from_file(file)
		else:
			self._weights = self.init_weights()
			self._params = self.init_params()		
			self._bn_params = self.init_bn_params()

			self.save_model("save")



	def train(self,model_inputs,val_inputs,lr,epochs,batch_size,print_every=20):

		run_id = str(np.random.randint(1000))

		print("\nTraining with run id:"  + run_id)
		writer = tf.summary.FileWriter('out.graph/run_' + run_id,flush_secs=30, graph=tf.get_default_graph())

		print("lr: " + str(lr))
		print("epochs: " + str(epochs))
		print("batch size: " + str(batch_size))

		# we need method called update weights
		i = 0

		epoch_size = int(model_inputs["x"].shape[0]/batch_size)
		inputs_x = model_inputs["x"]
		inputs_y = model_inputs["y"]

		shuffle_index =np.arange(inputs_x.shape[0])

		for e in range(epochs):
			# shuffle data
			shuffle_index =np.arange(inputs_x.shape[0])
			np.random.shuffle(shuffle_index)
			
			inputs_x = inputs_x[shuffle_index,...]
			inputs_y = inputs_y[shuffle_index,...]

			for b in range(epoch_size):
				batch_x = inputs_x[batch_size*b:batch_size*(b+1),...]
				batch_y = inputs_y[batch_size*b:batch_size*(b+1),...]

				# augment random images from batch_x
				#batch_x = augment(batch_x)

				batch_inputs = {"x":batch_x,"y":batch_y}
				cost, caches = self.forward_propagate(batch_inputs,self._weights,self._params,self._bn_params)
				gradients = self.backward_propagate(batch_inputs,caches)
				self.update_adam(gradients,i+1,lr)


				if i%print_every == 0:

					summary = tf.Summary(value=[tf.Summary.Value(tag='cost',simple_value=cost)])
					writer.add_summary(summary,i)

					validation_accuracy,_ = self.eval(val_inputs)

					accuracy_summary = tf.Summary(value=[tf.Summary.Value(tag='Validation Accuracy',simple_value=validation_accuracy)])
					writer.add_summary(accuracy_summary,i)

					data = [["Progress","Mini Batch Cost", "Validation Accuracy"],[str(int(b/float(epoch_size)*100)) + "% " + str(e) + "/" + str(epochs),cost,str(validation_accuracy) + "%"]]

					table = AsciiTable(data)
					table.title = "Stats run_" + run_id
	        		
					os.system('clear') 
					print(table.table)
					print("Printing every " + str(print_every) + " iterations") 

				i+=1


		print("Saving weights...")
		self.save_model("final_weights_"+run_id)





	def eval(self,inputs,batches=None):
		x = inputs['x']
		y = inputs['y']

		if batches is not None:

			epoch = int(y.shape[0]/batches)

			averages = np.array([])
			for b in range(epoch):

				batch_x = x[b*batches:(b+1)*batches,...]
				batch_y = y[b*batches:(b+1)*batches,...]
				batch_input = {"x":batch_x,"y":batch_y}

				cost, caches = self.forward_propagate(batch_input,self._weights,self._params,self._bn_params,run='test')

				accuracy = int(np.mean(batch_y.argmax(axis=1) == caches["A4"].argmax(axis=1)) * 100)
				averages = np.append(averages,accuracy)

			out = caches['A4']
			accuracy = np.mean(averages)
		else:
			cost, caches = self.forward_propagate(inputs,self._weights,self._params,self._bn_params,run='test')

			out = caches['A4']
			accuracy = int(np.mean(y.argmax(axis=1) == caches["A4"].argmax(axis=1)) * 100)

		return accuracy,out

	def update_adam(self,gradients,iteration,lr):
		beta = 0.9
		# repeat for each weight
		for key in self._weights:
			self._rms_velocity[key] = beta * self._rms_velocity[key] + ((1-beta) * np.square(gradients[key]))
			rms_corrected = self._rms_velocity[key]/(1-math.pow(beta,iteration))
			self._momentum_velocity[key] = beta * self._momentum_velocity[key] + ((1-beta) * gradients[key])
			momentum_corrected = self._momentum_velocity[key]/(1-math.pow(beta,iteration))
			self._weights[key] -= lr * momentum_corrected/np.sqrt(rms_corrected + 1e-8)

		for key in self._params:
			self._params[key] -= lr * gradients[key]


	def forward_propagate(self,model_inputs,weights,params,bn_params,run='train'):

		x = model_inputs["x"]
		y = model_inputs["y"]

		caches = {}

		Z1, caches["Z1"] = conv_fast(x,weights["W1"],weights["B1"],{'pad':1,'stride':1})
		
		BN1,bn_params["running_mu_1"],bn_params["running_sigma_1"],caches["BN1"] = batchnorm_forward(Z1,params["gamma1"],params["beta1"],bn_params["running_mu_1"],bn_params["running_sigma_1"],run)

		caches["A1"] = relu(BN1)

		Pool1, caches["Pool1"] = max_pooling(caches["A1"],2)
		
		Z2, caches["Z2"] = conv_fast(Pool1,weights["W2"],weights["B2"],{'pad':1,'stride':1})
		
		BN2,bn_params["running_mu_2"],bn_params["running_sigma_2"],caches["BN2"] = batchnorm_forward(Z2,params["gamma2"],params["beta2"],bn_params["running_mu_2"],bn_params["running_sigma_2"],run)

		caches["A2"] = relu(BN2)

		Pool2, caches["Pool2"] = max_pooling(caches["A2"],2)
		
		Z3, caches["Z3"] = conv_fast(Pool2,weights["W3"],weights["B3"],{'pad':1,'stride':1})

		BN3,bn_params["running_mu_3"],bn_params["running_sigma_3"],caches["BN3"] = batchnorm_forward(Z3,params["gamma3"],params["beta3"],bn_params["running_mu_3"],bn_params["running_sigma_3"],run)

		caches["A3"] = relu(BN3)

		Pool3, caches["Pool3"] = max_pooling(caches["A3"],2)
		
		pool3_reshape = Pool3.reshape(Pool3.shape[0],Pool3.shape[1] * Pool3.shape[2] * Pool3.shape[3])
		
		Z4, caches["Z4"] = fully_connected(pool3_reshape,weights["W4"],weights["B4"])

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

		dbn3,gradients["gamma3"],gradients["beta3"] = batchnorm_backward(dz3,caches["BN3"])

		dz3,gradients["W3"],gradients["B3"] = conv_fast_back(dbn3,caches["Z3"])
		da2 = max_pooling_back(dz3, caches["Pool2"])
		dz2 = relu_back(caches["A2"],da2)

		dbn2,gradients["gamma2"],gradients["beta2"] = batchnorm_backward(dz2,caches["BN2"])

		dz2,gradients["W2"],gradients["B2"] = conv_fast_back(dbn2,caches["Z2"])
		da1 = max_pooling_back(dz2, caches["Pool1"])
		dz1 = relu_back(caches["A1"],da1)
		dbn1,gradients["gamma1"],gradients["beta1"] = batchnorm_backward(dz1,caches["BN1"])

		dz1,gradients["W1"],gradients["B1"] = conv_fast_back(dbn1,caches["Z1"])

		return gradients


	def verify_gradients(self,inputs,verbose=True):

		cost, caches = self.forward_propagate(inputs,self._weights,self._params,self._bn_params)
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

		epsilon = 0.0000001

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

		cost1, caches1 = self.forward_propagate(inputs,weights1,self._params,self._bn_params)
		cost2, caches2 = self.forward_propagate(inputs,weights2,self._params,self._bn_params)

		return (cost1 - cost2) / (2. *epsilon)  

	def init_weights(self):
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

	def init_params(self):

		params = {}

		params["gamma1"] = np.ones(16)
		params["beta1"] = np.zeros(16)


		params["gamma2"] = np.ones(16)
		params["beta2"] = np.zeros(16)


		params["gamma3"] = np.ones(8)
		params["beta3"] = np.zeros(8)

		return params

	def init_bn_params(self):
		bn_params = {}

		bn_params["running_mu_1"] = np.zeros(16)
		bn_params["running_sigma_1"] = np.zeros(16)


		bn_params["running_mu_2"] = np.zeros(16)
		bn_params["running_sigma_2"] = np.zeros(16)


		bn_params["running_mu_3"] = np.zeros(8)
		bn_params["running_sigma_3"] = np.zeros(8)

		return bn_params


	def save_model(self,filename):
		model = {}

		for weight in self._weights:
			model[weight] = self._weights[weight]

		for param in self._params:
			model[param] = self._params[param]

		for bn in self._bn_params:
			model[bn] = self._bn_params[bn]


		np.save(filename + '.npy',model)
		print("Successfully saved weights")



	def load_model_from_file(self,filename):
		weights = {}
		params = {}
		bn_params = {}

		data_load = np.load(filename + '.npy').item()


		weights["W1"] = data_load.get('W1')
		weights["W2"] = data_load.get('W2')
		weights["W3"] = data_load.get('W3')
		weights["W4"] = data_load.get('W4')
	    
		weights["B1"] = data_load.get('B1')
		weights["B2"] = data_load.get('B2')
		weights["B3"] = data_load.get('B3')
		weights["B4"] = data_load.get('B4')


		params["gamma1"] = data_load.get('gamma1')
		params["beta1"] = data_load.get('beta1')

		params["gamma2"] = data_load.get('gamma2')
		params["beta2"] = data_load.get('beta2')

		params["gamma3"] = data_load.get('gamma3')
		params["beta3"] = data_load.get('beta3')


		bn_params["running_mu_1"] = data_load.get('running_mu_1')
		bn_params["running_sigma_1"] = data_load.get('running_sigma_1')

		bn_params["running_mu_2"] = data_load.get('running_mu_2')
		bn_params["running_sigma_2"] = data_load.get('running_sigma_2')

		bn_params["running_mu_3"] = data_load.get('running_mu_3')
		bn_params["running_sigma_3"] = data_load.get('running_sigma_3')

		for key in weights:
			self._rms_velocity[key] = 0
			self._momentum_velocity[key] = 0


		return weights,params,bn_params

