import numpy as np
from cifar_loader import cifar10
from solver.solvers import CNN


def main():

	cifar10.maybe_download_and_extract()
	train_x_raw, train_y, train_y_one_hot = cifar10.load_training_data()
	test_x_raw, test_y, test_y_one_hot = cifar10.load_test_data()
	classes = cifar10.load_class_names()

	# set validation
	train_x_validation = train_x_raw[:1000,...]
	train_y_validation = train_y[:1000,...]
	train_y_one_hot_validation = train_y_one_hot[:1000,...]

	# set train
	train_x_raw = train_x_raw[0:49000,...]
	train_y = train_y[0:49000,...]
	train_y_one_hot=train_y_one_hot[0:49000,...]

	# gather mean
	train_mean = np.mean(train_x_raw )
	train_max = np.max(train_x_raw )
	train_min = np.min(train_x_raw )

	# normalize
	train_x_raw  = (train_x_raw - train_mean)/train_max-train_min
	train_x_validation = (train_x_validation - train_mean)/train_max-train_min

	# init cnn
	cnn_1 = CNN()	

	# Check gradients
	grad_approx_inputs={'x':train_x_raw[0:2,...],'y':train_y_one_hot[0:2,...]}
	cnn_1.verify_gradients(grad_approx_inputs,True)

	# Format data
	train_inputs={'x':train_x_raw,'y':train_y_one_hot}
	val_inputs={'x':train_x_validation,'y':train_y_one_hot_validation}

	# Train
	cnn_1.train(train_inputs,val_inputs,0.001,epochs=5,batch_size=32)

	cnn_1.test(val_inputs)


if __name__ == "__main__":
	main()