'''
	Data generator inheritting from the Sequence Keras Generator
'''
from keras.utils import Sequence
import numpy as np
import cv2
from augmentation import augment
from glob import glob
import os
from matplotlib import pyplot as plt
from math import ceil, floor
from skimage import exposure

from config import IMAGE_PATHS, MASK_PATHS
from PIL import Image, ImageOps
from keras.preprocessing.image import img_to_array, Iterator
from keras.preprocessing.image import load_img as load_img

class Generator(Sequence):
	def __init__(self, x_set, y_set, batch_size, target_shape=(256,256), shuffle=False):
		'''
			x_set: the paths to load images
			y_set: the path to load annotations
			NOTE: set_x and set_y should be aligned
			batch_size: the size of the batch for each training
		'''
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size
		self.target_shape = target_shape
		self.shuffle = shuffle
		self.on_epoch_end()

	def on_epoch_end(self):
		print("\n Epoch ended")
		pass

	def _data_generation (self, images_to_load, corresponding_masks, shape):
		X = list(map(lambda path: cv2.imread(path), images_to_load))		
		
		'''
		X = []
		for path in images_to_load:
			X.append(img_to_array(load_img(path, target_size=(256, 256), grayscale=True)) / 255)
		'''
		
		y = list(map(lambda path: cv2.imread(path), corresponding_masks))
		
		'''
		y = []
		for path in corresponding_masks:
			y.append(img_to_array(load_img(path, target_size=(256, 256), grayscale=True )) / 255)
		'''

		for i, img in enumerate(y):
			if img is None:
				print(' [EROOR INFO]  None image Loaded:  ', corresponding_masks[i])
				exit(1)

		X,y = augment(X, y)
		X = list(map(lambda img: cv2.resize(img, shape) if img.shape!=shape else img, X))
		y = list(map(lambda img: cv2.resize(img, shape) if img.shape!=shape else img, y))
		


		X = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), X))
		y = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), y))

		# normalize input images and mask images
		X = list(map(lambda x: exposure.equalize_adapthist(x), X))
		#X = list(map(lambda x: x/255, X))
		y = list(map(lambda x: x/255, y))

		# expand dims to fit the network architecture
		X = list(map(lambda x: np.expand_dims(x,3), X))
		y = list(map(lambda x: np.expand_dims(x,3), y))

		return X,y
	
	def __len__(self):
		# returns the number of batches that should generated to traverse the complete set
		return int(np.ceil(len(self.x) / float(self.batch_size)))
	

	def __getitem__(self, idx):
		image_paths = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		mask_paths = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
		X, y  = self._data_generation(image_paths, mask_paths, self.target_shape)
		X = np.array(X)
		y = np.array(y)
		return  X, y

def createGens(batch_size = 32, split = 0.2):
	image_paths = glob(os.path.join(IMAGE_PATHS, "*.png"))
	mask_paths = glob(os.path.join(MASK_PATHS, "*_mask.png"))

	if len(image_paths) ==0:
		print ("[ERROR] Image Paths are an empty List")
		exit(1)
	if len(mask_paths) ==0:
		print ("[ERROR] Image Paths are an empty List")
		exit(1)
	
	image_ids = list(map(lambda x: x.split('/')[-1].split('.')[0], image_paths ))
	mask_ids = list(map(lambda x:  x.split('/')[-1].split('_mask')[0], mask_paths ))

	intersection = np.intersect1d(image_ids, mask_ids)
	image_paths = list(map(lambda x: os.path.join(IMAGE_PATHS, x+".png",), intersection))
	annotation_paths = list(map(lambda x: os.path.join(MASK_PATHS, x + "_mask.png"), intersection))
	
	train_paths = image_paths[0: floor((1-split)* len(image_paths))]
	train_annotation_paths = annotation_paths[0: floor((1-split)* len(image_paths))]

	validation_paths = image_paths[floor((1-split)* len(image_paths)) : ]
	validation_annotation_paths =  annotation_paths[floor((1-split)* len(image_paths)) :]

	return Generator(train_paths, train_annotation_paths, batch_size), Generator(validation_paths, validation_annotation_paths, batch_size)

if __name__=="__main__":
	print("[INFO] Testing sequence generator")
	image_paths = glob(os.path.join(IMAGE_PATHS, "*.png"))

	if len(image_paths) ==0:
		print ("[ERROR] Image Paths are an empty List")
		exit(1)
	intersection = list(map(lambda x: x.split('/')[-1].split('.')[0], image_paths))
	annotation_paths = list(map(lambda x: os.path.join(MASK_PATHS, x + "_mask.png"), intersection))

	trainGen, valGen = createGens()

	count = 0
	for X,y in trainGen:
		print(X[0].shape)
		print(y[0].shape)
		count +=1
		if count>3:
			break

	'''
	plt.subplot(221)
	plt.imshow(X[0].squeeze())
	plt.subplot(222)
	plt.imshow(y[0].squeeze())

	plt.subplot(223)
	plt.imshow(X[1].squeeze())
	plt.subplot(224)
	plt.imshow(y[1].squeeze())

	print(X[0].min())
	print(X[0].max())
	print(X[0].shape)
	plt.show()
	'''