import numpy as np
from cifar_loader import cifar10
from solver.solvers import CNN


def main():

	# 1. create test training and cv data
	# 2. initialize cnn
	# 3. set layers of cnn
	# 4. train

	cifar10.maybe_download_and_extract()
	train_x_raw, train_y, train_y_one_hot = cifar10.load_training_data()
	test_x_raw, test_y, test_y_one_hot = cifar10.load_test_data()
	classes = cifar10.load_class_names()

	train_x_validation = train_x_raw[:1000,...]
	train_y_validation = train_y[:1000,...]
	train_y_one_hot_validation = train_y_one_hot[:1000,...]

	train_x_raw = train_x_raw[0:49000,...]
	train_y = train_y[0:49000,...]
	train_y_one_hot=train_y_one_hot[0:49000,...]

	train_x_raw = train_x_raw[0:100,...]
	train_y = train_y[0:100,...]
	train_y_one_hot = train_y_one_hot[0:100,...]

	train_mean = np.mean(train_x_raw )
	train_max = np.max(train_x_raw )
	train_min = np.min(train_x_raw )

	train_x_raw  = (train_x_raw - train_mean)/train_max-train_min


	cnn_1 = CNN()	
	cnn_1.train(model_inputs={'x':train_x_raw,'y':train_y},lr=1e-3,epochs=5,batch_size=32)


if __name__ == "__main__":
	main()