PROJECT_ROOT = "." #TODO: add the project root directory here ...

DESTINATION = "localhost"

#TODO: Hint --> instead of image directories masks etc.
# do provide an input and an output instance example 
# the model will infer what is going on with the way that 
# are organized

if DESTINATION=="hpc": #hpc setup
    INPUT_EXAMPLE = \
        '/users/pa17/candrik/isic_challenge_2018_task_2/ISIC2018_Task1-2_Training_Input/ISIC_0014933.jpg'
    OUTPUT_EXAMPLE = \
        "/users/pa17/candrik/isic_challenge_2018_task_2/ISIC2018_Task2_Training_GroundTruth_v3/ISIC_0014933_attribute_pigment_network.png"
    MASK_PATHS = \
        "/users/pa17/candrik/unet_on_isic/preprocessing/outputs.json" # # this import specific paths for images -- leave empty marks if does not #TODO: exist
    PREPROPROCESSING_MASK_EXAMPLE = \
        "/users/pa17/candrik/task_1/ISIC2018_Task1_Training_GroundTruth/ISIC_0014933_segmentation.png" # this to apply a preprocessing masking over input image. leave empty if it is not required
    EPOCHS = 60
    BATCH_SIZE = 32
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 4
    GPU_NO = 2
    WORKERS = 1
    MULTIPROCESSING = False
    PRETRAINED_WEIGHTS = None # offer the pretrained weights path here
    WEIGHTS_FILE = "isic_task_1_unet_pigment_network.h5.h5"
else: # localhost setup
    INPUT_EXAMPLE = \
        '/Users/candrikos/python_workspace/isic_task_2/data/ISIC2018_Task1-2_Training_Input_sample/ISIC_0014933.jpg'
    OUTPUT_EXAMPLE = \
        "/Users/candrikos/python_workspace/isic_task_2/data/ISIC2018_Task2_Training_GroundTruth_v3_sample/ISIC_0014933_attribute_globules.png"
    MASK_PATHS = \
        "/Users/candrikos/python_workspace/isic_preprocessing/outputs.json" # # this import specific paths for images -- leave empty marks if does not #TODO: exist
    PREPROPROCESSING_MASK_EXAMPLE = \
        "/Users/candrikos/python_workspace/isic_task_2/data/ISIC2018_Task1_Training_GroundTruth_sample/ISIC_0014933_segmentation.png" # this to apply a preprocessing masking over input image. leave empty if it is not required
    EPOCHS = 4
    BATCH_SIZE = 32
    GPU_NO = 1
    WORKERS = 1
    STEPS_PER_EPOCH = 2
    VALIDATION_STEPS = 1
    MULTIPROCESSING = False
    PRETRAINED_WEIGHTS = None # offer the pretrained weights path here
    WEIGHTS_FILE = "isic_task_1_unet_pigment_network.h5"

# this is the accuracy you need to rich ... early stopping will be applied ... required for transfer learning
THRESHOLD_ACCURACY = 0.8
MIN_EPOCHS = 1
EARLY_STOPPING = False


DEBUG_LEVEL = 1
