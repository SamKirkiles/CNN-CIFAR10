from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

def augment(input):
	seq = iaa.Sequential([
	    iaa.Fliplr(0.5)
	])

	augmented = seq.augment_images((input*255).astype(np.uint8))

	return (augmented/255.).astype(np.float64)

