'''
    Genrates data to be laoded while training Unet for chest segmentation
'''

import os
from glob import glob

import cv2 #FIXME: this should be removed die to GIL parallelizing problem

import numpy as np
from threadSafeGen import threadsafe_generator
from math import floor, ceil
from augmentation import augment
from config import INPUT_EXAMPLE, OUTPUT_EXAMPLE # laod images and masks from a particular directory
from skimage import exposure
from utils import generate_paths


print(INPUT_EXAMPLE)

@threadsafe_generator
def Generator(image_paths, mask_paths, shape=(256,256), batch_size = 32):
    print("The len of image paths is: ", len(image_paths))
    batch_size = batch_size if batch_size < len(image_paths) else len(image_paths)
    idx = 0
    while True:
        images_to_load = image_paths[idx*batch_size:idx*batch_size+batch_size]
        corresponding_masks = mask_paths[idx*batch_size:idx*batch_size+batch_size]

        X = list(map(lambda path: cv2.imread(path), images_to_load))
        y = list(map(lambda path: cv2.imread(path), corresponding_masks))

        for i, img in enumerate(y):
            if img is None:
                print(' [EROOR INFO]  None image Loaded:  ', corresponding_masks[i])
                exit(1)

        X = list(map(lambda img: cv2.resize(img, shape) if img.shape!=shape else img, X))
        y = list(map(lambda img: cv2.resize(img, shape) if img.shape!=shape else img, y))

        X, y = augment(X, y)

        X = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), X))
        y = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), y))
        
        X = list(map(lambda img: exposure.equalize_hist(img), X))
        y = list(map(lambda img: img//255, y))

        # expand dims to fit the network architecture
        X = list(map(lambda x: np.expand_dims(x,2), X))
        y = list(map(lambda x: np.expand_dims(x,2), y))
        if len(image_paths) < batch_size:
            batch_size = len(image_paths)
        print("This is the id ", idx)
        print("The batch size is: ", batch_size)
        print("The len of image_paths is: ", len(image_paths))
        idx = (idx + 1) % (len(image_paths) // batch_size)
        yield np.array(X), np.array(y)

def createGens(input_example=INPUT_EXAMPLE, output_example=OUTPUT_EXAMPLE, shape=(256,256), batch_size = 32, split=0.2):
    '''
        Denotes and returns Train and Validation genrators ...
    '''
    input_root, input_suffix, output_root, output_suffix, output_separator = generate_paths(input_example, output_example)
    
    print("Input: ", input_root)
    print("Output root: ", output_root)

    image_paths = glob(os.path.join(input_root, "*" + input_suffix))
    print("-- #Image_paths: ", len(image_paths) )
    mask_paths = glob(os.path.join(output_root, "*" + output_suffix))
    print("-- #Mask_paths: ", len(mask_paths))
    print(image_paths)
    print(mask_paths)
    # apparently not all images have a corresponding mask so we need the intersection of those
    image_ids = list(map(lambda x: x.split('/')[-1].split(input_suffix)[0], image_paths))
    mask_ids = list(map(lambda x: x.split('/')[-1].split(output_suffix)[0], mask_paths))

    intersection = list(set(image_ids) & set(mask_ids))

    image_paths = list(map(lambda x: os.path.join(input_root, x + input_suffix), intersection))
    mask_paths = list(map(lambda x: os.path.join(output_root, x + output_suffix), intersection)) # output_suffix is whatever is placed after the id ...  
    print("This is the intersection", len(intersection))

    dataset_len = len(image_paths)
    train_image_paths = image_paths[:floor(dataset_len * (1 - split))]
    train_mask_paths = mask_paths[:floor(dataset_len * (1 - split))]

    validation_image_paths = image_paths[ceil(dataset_len*(1 - split)):]
    validation_mask_paths = mask_paths[ceil(dataset_len*(1 - split)):]
    
    return Generator(train_image_paths, train_mask_paths, shape, batch_size), Generator(validation_image_paths, validation_mask_paths, shape, batch_size)


if __name__ == "__main__":
    k=0
    trainGen, valGen = createGens(batch_size=2)

    for X, y in trainGen:
        print("TrainGen: ")
        print("The X shape, y shape is:")
        print(X.shape, y.shape)
        print(X[0].shape)
        print(X[1].shape)
        k+=1
        if k > 2:
            break

    for X, y in valGen:
        print("ValGen: ")
        print(X.shape, y.shape)
        print(X[0].shape)
        #plot_boxes(X[0], [y[0]])
        k+=1
        if k > 2:
            break
