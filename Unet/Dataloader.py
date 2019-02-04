'''
    Genrates data to be laoded while training Unet for chest segmentation
'''

import os
from glob import glob

import cv2 #FIXME: this should be removed die to GIL parallelizing problem #FIXME: this is the first one that should take place tomorrow morining

import numpy as np
from threadSafeGen import threadsafe_generator
from math import floor, ceil
from augmentation import augment
from config import INPUT_EXAMPLE, OUTPUT_EXAMPLE, MASK_PATHS, PREPROPROCESSING_MASK_EXAMPLE, DEBUG_LEVEL # laod images and masks from a particular directory
from skimage import exposure
from utils import generate_paths, resize_img, load_image, get_segment_crop

import json # in case of loading particular mask paths from som json

@threadsafe_generator
def Generator(image_paths, mask_paths, shape=(256,256), batch_size = 32, \
    preprocessing_mask_paths=None):
    # ================ Debugging routine follows =============
    if DEBUG_LEVEL >= 2:
        assert image_paths, \
            "Empty list of image paths provided in Generator"
        assert mask_paths, \
            "Empty list od mask_paths provided in Generator"
    # ========================================================
    batch_size = batch_size if batch_size < len(image_paths) else len(image_paths)
    idx = 0
    while True:
        images_to_load = image_paths[idx*batch_size:idx*batch_size+batch_size]
        corresponding_masks = mask_paths[idx*batch_size:idx*batch_size+batch_size]
        
        X = list(map(lambda path: cv2.imread(path), images_to_load)) #TODO: this should be updated to a no cv2 solution
        y = list(map(lambda path: cv2.imread(path), corresponding_masks))

        # ================ Debugging routine follows =============
        if DEBUG_LEVEL >= 2:
            X_shapes = list(map(lambda x: x.shape, X))
            y_shapes = list(map(lambda x: x.shape, y))
            assert X_shapes == y_shapes
        # ========================================================
        if preprocessing_mask_paths:
            preprocessing_masks_to_load = preprocessing_mask_paths[idx*batch_size:idx*batch_size+batch_size]
            preprocessing_masks = list(map(lambda path: cv2.imread(path), preprocessing_masks_to_load))
            
            # ================ Debugging routine follows =============
            if DEBUG_LEVEL>=2:
                preprocess_mask_shapes = list(map(lambda x: x.shape, \
                    preprocessing_masks))
                assert preprocess_mask_shapes == X_shapes, \
                    "Preprocessing masks and images do not feature the same shape"
            # ========================================================

            l = list(zip(X, preprocessing_masks))

            # do segment the X,y with the preprocessing mask
            X = list(map(lambda el: get_segment_crop(el[0], mask=el[1]), l))
            l = zip(y, preprocessing_masks)
            y = list(map(lambda el: get_segment_crop(el[0], mask=el[1]), l))
            # FIXME: set the shape here ... 
            X = list(map(lambda img: resize_img(img), X))
            y = list(map(lambda mask: resize_img(mask), y))

        for i, img in enumerate(y):
            if img is None:
                print(' [EROOR INFO]  None image Loaded:  ', corresponding_masks[i])
                exit(1)

        #X = list(map(lambda img: cv2.resize(img, shape) if img.shape!=shape else img, X))
        #y = list(map(lambda img: cv2.resize(img, shape) if img.shape!=shape else img, y))

        #X, y = augment(X, y) #FIXME: this is an issue ... 

        X = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), X))
        y = list(map(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), y))
        
        X = list(map(lambda img: exposure.equalize_hist(img), X))
        y = list(map(lambda img: img//255, y))

        # expand dims to fit the network architecture
        X = list(map(lambda x: np.expand_dims(x,2), X))
        y = list(map(lambda x: np.expand_dims(x,2), y))
        if len(image_paths) < batch_size:
            batch_size = len(image_paths)
        
        # ================ Debugging routine follows =============
        if DEBUG_LEVEL > 1:
            assert image_paths, "Empty image_paths list"
        if DEBUG_LEVEL > 0:
            assert batch_size, "Batch_size is 0 really?"
        # ========================================================

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
    print("-- #Image_paths: ", len(image_paths))
    mask_paths = glob(os.path.join(output_root, "*" + output_suffix))
    print("-- #Mask_paths: ", len(mask_paths))

    if MASK_PATHS!="": # update mask paths to the dedicated ones if it is denoted ...
        with open(MASK_PATHS, 'r') as f:
            dedicated_mask_paths = json.load(f)
        attribute = OUTPUT_EXAMPLE.split(output_separator)[-1].split('.')[0]
        mask_paths = dedicated_mask_paths[attribute]
        print(" -- However we gonna consume only ", len(mask_paths))
    
    if PREPROPROCESSING_MASK_EXAMPLE:
        # TODO: we care only about preprocessing_output and suffix
        _, _, preprocessing_root, preprocessing_suffix, _ = generate_paths \
            (INPUT_EXAMPLE, PREPROPROCESSING_MASK_EXAMPLE)
        preprocessing_paths = glob(os.path.join(preprocessing_root, "*"+\
            preprocessing_suffix))
        preprocessing_ids = list(map(lambda  path: path.split('/')[-1]\
            .split(preprocessing_suffix)[0], \
                preprocessing_paths))
    else:
        preprocessing_paths = []

    # apparently not all images have a corresponding mask so we need the intersection of those
    image_ids = list(map(lambda x: x.split('/')[-1].split(input_suffix)[0], \
         image_paths))    
    mask_ids = list(map(lambda x: x.split('/')[-1].split(output_suffix)[0], \
         mask_paths))
    if preprocessing_paths:
        intersection = list(set(image_ids) & set(mask_ids) & \
             set(preprocessing_ids))
        preprocessing_paths = list(map(lambda x: os.path.join( \
            preprocessing_root, x+preprocessing_suffix), \
                intersection))
    else:
        intersection = list(set(image_ids) & set(mask_ids))

    image_paths = list(map(lambda x: os.path.join( \
        input_root, x + input_suffix), \
            intersection))
    mask_paths = list(map(lambda x: os.path.join( \
        output_root, x + output_suffix), \
            intersection)) # output_suffix is whatever is placed after the id ...

    dataset_len = len(image_paths)
    train_image_paths = image_paths[:floor(dataset_len * (1 - split))]
    train_mask_paths = mask_paths[:floor(dataset_len * (1 - split))]
    if preprocessing_paths:
        train_preprocessing_paths = preprocessing_paths[:floor(dataset_len * (1 - split))]
    else:
        train_preprocessing_paths = []

    validation_image_paths = image_paths[ceil(dataset_len*(1 - split)):]
    validation_mask_paths = mask_paths[ceil(dataset_len*(1 - split)):]
    if preprocessing_paths:
        validation_preprocessing_paths = preprocessing_paths[ceil(dataset_len*(1 - split)):]
    else:
        validation_preprocessing_paths = []

    return Generator(train_image_paths, train_mask_paths, shape, batch_size, train_preprocessing_paths), Generator(validation_image_paths, validation_mask_paths, shape, batch_size, validation_preprocessing_paths)


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
    '''
    for X, y in valGen:
        print("ValGen: ")
        print(X.shape, y.shape)
        print(X[0].shape)
        #plot_boxes(X[0], [y[0]])
        k+=1
        if k > 2:
            break
    '''
