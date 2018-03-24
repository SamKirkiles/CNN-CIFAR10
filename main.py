import numpy as np
from cifar_loader import cifar10
from solver.solvers import CNN
import atexit
import matplotlib.pyplot as plt

def exit_handler():
    print("Saving weights...")
    print(weights["W1"][0,0,0,0])
    np.save('train_weights.npy',weights)


def main():

	train = True

	# Set weights to the name of the file with compatible weights or None to train from scratch
	weights = "trained_weights_val70"

	# Load data 
	cifar10.maybe_download_and_extract()
	train_x_raw, train_y, train_y_one_hot = cifar10.load_training_data()
	test_x_raw, test_y, test_y_one_hot = cifar10.load_test_data()
	classes = cifar10.load_class_names()

	# Create validation set
	train_x_validation = train_x_raw[:1000,...]
	train_y_validation = train_y[:1000,...]
	train_y_one_hot_validation = train_y_one_hot[:1000,...]

	# Create Test set
	train_x_raw = train_x_raw[0:49000,...]
	train_y = train_y[0:49000,...]
	train_y_one_hot=train_y_one_hot[0:49000,...]

	# Normalization stats
	train_mean = np.mean(train_x_raw )
	train_max = np.max(train_x_raw )
	train_min = np.min(train_x_raw )

	# Normalize
	train_x_raw  = (train_x_raw - train_mean)/(train_max-train_min)
	train_x_validation = (train_x_validation - train_mean)/(train_max-train_min)

	# Initialize CNN

	if train:
		# Initialize CNN
		cnn_1 = CNN(file=weights)	
		# Create file name to save weights to if model crashes
		atexit.register(cnn_1.save_model,"WEIGHT_DUMP")

		# Check gradients
		grad_approx_inputs={'x':train_x_raw[0:2,...],'y':train_y_one_hot[0:2,...]}
		cnn_1.verify_gradients(grad_approx_inputs,True)

		# Format data
		train_inputs={'x':train_x_raw,'y':train_y_one_hot}
		val_inputs={'x':train_x_validation,'y':train_y_one_hot_validation}

		cnn_1.train(train_inputs,val_inputs,0.0005,epochs=10,batch_size=32)

	else:

		cnn_1 = CNN(file="trained_weights_val70")	

		train_inputs={'x':train_x_raw,'y':train_y_one_hot}
		val_inputs={'x':train_x_validation,'y':train_y_one_hot_validation}

		validation_accuracy,results = cnn_1.eval(val_inputs)
		print("Validation Accuracy: " + str(validation_accuracy) + "%")

		train_x_validation = (train_x_validation + train_mean)*(train_max-train_min)

		graph_inputs={'x':train_x_validation,'y':train_y_one_hot_validation,'yhat':results}

		graph_results(graph_inputs,classes)

		# Uncoment for test accuracy (time consuming 10k examples)
		"""
		test_inputs = {"x":test_x_raw,"y":test_y_one_hot}
		print("Testing... This could take a while...")
		test_accuracy = cnn_1.eval(test_inputs,batches=50)
		print("Test Accuracy: " + str(test_accuracy) + "%")
		"""

def graph_results(inputs,classes,num=20):
	input_x = inputs['x']
	input_y = inputs['y']
	yhat = inputs['yhat']

	for img in range(num):
		fig = plt.figure(1)
		fig.add_subplot(121)
		plt.imshow(input_x[img,...])
		fig.add_subplot(122)
		y = yhat[img,...]
		x = [0,1,2,3,4,5,6,7,8,9]
		plt.yticks(np.arange(10), classes)
		plt.barh(x,y)
		plt.show()



if __name__ == "__main__":

	main()