'''
    Basic stochastic image augmentation
'''

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import cv2
import os

from config import

ia.seed(1)
def augment(image_list, mask_list):

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.9, aug)

    # Define our sequence of augmentation steps that will be applied to every image.
    seq = iaa.Sequential(
        [
            #
            # Apply the following augmenters to most images.
            #
            iaa.Fliplr(0.5, name ="horizontal_flip"), # horizontally flip 50% of all images
            iaa.Flipud(0.2, name = "vertical_flip"), # vertically flip 20% of all images

            # crop some of the images by 0-10% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.1), name="crop" )),

            # Apply affine transformations to some of the images
            # - scale to 80-120% of image height/width (each axis independently)
            # - translate by -20 to +20 relative to height/width (per axis)
            # - rotate by -45 to +45 degrees
            # - shear by -16 to +16 degrees
            # - order: use nearest neighbour or bilinear interpolation (fast)
            # - mode: use any available mode to fill newly created pixels
            #         see API or scikit-image for which modes are available
            # - cval: if the mode is constant, then use a random brightness
            #         for the newly created pixels (e.g. sometimes black,
            #         sometimes white)
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL,
                name = "basic_affine"
            )),

            #
            # Execute 0 to 5 of the following (less important) augmenters per
            # image. Don't execute all of them, as that would often be way too
            # strong.
            #
            iaa.SomeOf((0, 5),
                [
                    # Convert some images into their superpixel representation,
                    # sample between 20 and 200 superpixels per image, but do
                    # not replace all superpixels with their average, only
                    # some of them (p_replace).
                    sometimes(
                        iaa.Superpixels(
                            p_replace=(0, 1.0),
                            n_segments=(20, 200),
                            name = "superpixeling"
                        )
                    ),

                    # Blur each image with varying strength using
                    # gaussian blur (sigma between 0 and 3.0),
                    # average/uniform blur (kernel size between 2x2 and 7x7)
                    # median blur (kernel size between 3x3 and 11x11).
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11))
                    ],
                        name = "blur"
                    ),

                    # Sharpen each image, overlay the result with the original
                    # image using an alpha between 0 (no sharpening) and 1
                    # (full sharpening effect).
                    iaa.Sharpen(
                        alpha=(0, 1.0), lightness=(0.75, 1.5),
                        name = "sharpen"
                        ),

                    # Same as sharpen, but for an embossing effect.
                    iaa.Emboss(
                        alpha=(0, 1.0), strength=(0, 2.0),
                        name = "emboss"
                    ),

                    # Search in some images either for all edges or for
                    # directed edges. These edges are then marked in a black
                    # and white image and overlayed with the original image
                    # using an alpha of 0 to 0.7.
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.7)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.7), direction=(0.0, 1.0)
                        ),
                    ],
                    name = "edge_sharpenning"
                    )),

                    # Add gaussian noise to some images.
                    # In 50% of these cases, the noise is randomly sampled per
                    # channel and pixel.
                    # In the other 50% of all cases it is sampled once per
                    # pixel (i.e. brightness change).
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5,
                        name = "noise"
                    ),

                    # Either drop randomly 1 to 10% of all pixels (i.e. set
                    # them to black) or drop them on an image with 2-5% percent
                    # of the original size, leading to large dropped
                    # rectangles.
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.05),
                            per_channel=0.2
                        ),
                    ], name="drop"),

                    # Invert each image's chanell with 5% probability.
                    # This sets each pixel value v to 255-v.
                    iaa.Invert(0.05, per_channel=True, name="invert"), # invert color channels

                    # Add a value of -10 to 10 to each pixel.
                    iaa.Add((-10, 10), per_channel=0.5, name="random_add"),

                    # Change brightness of images (50-150% of original value).
                    iaa.Multiply((0.5, 1.5), per_channel=0.5, name = "multiply"),

                    # Improve or worsen the contrast of images.
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5, name="cotrastNormalization"),

                    # Convert each image to grayscale and then overlay the
                    # result with the original with random alpha. I.e. remove
                    # colors with varying strengths.
                    iaa.Grayscale(alpha=(0.0, 1.0), name="grayScale"),

                    # In some images move pixels locally around (with random
                    # strengths).
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25, name="elasticTransformations")
                    ),

                    # In some images distort local areas with varying strength.
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05), name="localAffine"))
                ],
                # do all of the above augmentations in random order,
                random_order=True
            )
        ],
        # do all of the above augmentations in random order
        random_order=True
    )

    noOps = [
        "superpixeling",
        "blur",
        "emboss",
        "sharpen",
        "edge_sharpenning",
        "noise",
        "drop",
        "invert",
        "random_add",
        "multiply",
        "cotrastNormalization",
        "grayScale",
        "elasticTransformations",
        "localAffine"
    ]

    def activator(images, augmenter, parents, default):
        return False if augmenter.name in noOps else default
    
    seq_det = seq.to_deterministic()
    images_aug = seq_det.augment_images(image_list)
    masks_aug = seq_det.augment_images( mask_list, hooks=ia.HooksImages(activator=activator) )

    masks_aug = list(map(lambda mask: cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1], list(masks_aug)))

    return images_aug, masks_aug


images = "/Users/candrikos/python_workspace/lungs_segmentation/images"
masks = "/Users/candrikos/python_workspace/lungs_segmentation/masks"

if __name__=="__main__":
    print ("[INFO] testing the augmentation process")
    image_paths = glob(os.path.join(images, "*.png"))
    mask_paths = list(map(lambda image_path: os.path.join(masks, image_path.split("/")[-1].split(".")[0]+"_mask.png"), image_paths ))
    
    images = list(map(lambda path: cv2.imread(path), image_paths ))
    masks =  list(map(lambda path: cv2.imread(path), mask_paths))
    for i, _ in enumerate(images):
        print (images[i].shape)

    images = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), images))
    masks = list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), masks))

    images, masks = augment(images, masks)

    for i, _ in enumerate(images):
        print (images[i].shape)

    j = 0
    plots_num = len(images)
    
    print(masks[0])
    print(plots_num)

    for i, _ in enumerate(images):
        plt.subplot(plots_num,2,j+1)
        plt.imshow(images[i], 'gray')
        plt.subplot(plots_num,2,j+2)
        plt.imshow(masks[i], 'gray')
        j+=2
    plt.show()
    

    print(masks)