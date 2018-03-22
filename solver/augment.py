from imgaug import augmenters as iaa
import imgaug as ia
import numpy as np

def augment(input):
	seq = iaa.Sequential([
	    iaa.Fliplr(0.5), 
	    iaa.Sometimes(0.2,iaa.Affine(
	    	rotate=(-45,45),
	    	mode="symmetric"),iaa.Noop(),
	    ),
	    iaa.Sometimes(0.05,iaa.PiecewiseAffine(scale=(0.01, 0.05),mode="symmetric"),iaa.Noop(),
	    )
	])

	augmented = seq.augment_images((input*255).astype(np.uint8))

	return (augmented/255.).astype(np.float64)

