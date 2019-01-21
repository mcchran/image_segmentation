from matplotlib import pyplot as plt
from matplotlib import patches
from src.myModels import Unet as Model
import cv2
from skimage import exposure
from scipy import ndimage 
import numpy as np
import sys
import math

def load_model():
	model = Model()
	model.load_weights("../weights/lungs_1.h5", by_name = True)
	return model

def get_mask(img, predictor):
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	grey = exposure.equalize_adapthist(grey)
	grey = cv2.resize(grey, (256,256)) 
	grey = np.expand_dims(grey,0)
	grey = np.expand_dims(grey, 3)
	preds = predictor.predict(grey)
	mask = preds > 0.5
	return mask.squeeze()

def get_boxes(mask):
	labeled_image, features = ndimage.label(mask.squeeze().astype(int))
	objs = ndimage.find_objects(labeled_image)
	obs = []
	for ob in objs:                                                    
		obs.append([ob[1].start, ob[0].start, ob[1].stop, ob[0].stop])
	return obs

def plot_boxes(img, objects):
	fig,ax = plt.subplots(1)
	ax.imshow(img.squeeze(), cmap="gray")
	rects = []
	for ob in objects:
		rects.append( patches.Rectangle( (ob[0], ob[1]), ob[2] - ob[0], ob[3]-ob[1], linewidth=1,edgecolor='r',facecolor='none'))
	while len(rects)>0:
		ax.add_patch(rects.pop())
	plt.savefig("test_squares.png")
	plt.show()

def clean_margins(img):
	grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_OTSU)
	bbox = cv2.boundingRect(thresholded)
	x, y, w, h = bbox
	foreground = img[y:y+h, x:x+w]
	return foreground

def clean_angles(img):
	img_before = img
	img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
	img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
	lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

	angles = []

	for x1, y1, x2, y2 in lines[0]:
		cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
		angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
		angles.append(angle)

		median_angle = np.median(angles)
		img_rotated = ndimage.rotate(img_before, median_angle)
		print("Angle is ", median_angle)

		return img_rotated

if __name__=="__main__":
	if len(sys.argv) < 2:
		print("[INFO] make sure you provide a CXR image path!")
		exit(1)
	img = cv2.imread(sys.argv[1])
	#img = clean_angles(img)
	plt.imshow(img)
	plt.show()
	
	model = load_model()
	mask = get_mask(img, model)
	plt.imshow(mask)
	plt.show()
	plot_boxes(cv2.resize(img, mask.shape), get_boxes(mask))
