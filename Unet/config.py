PROJECT_ROOT = "." #TODO: add the project root directory here ...

DESTINATION = "localhost"

#TODO: Hint --> instead of image directories masks etc.
# do provide an input and an output instance example 
# the model will infer what is going on with the way that 
# are organized

if DESTINATION=="hpc":
    INPUT_EXAMPLE = '../data/ChinaSet_AllFiles/CXR_png'
    OUTPUT_EXAMPLE = '../data/mask'
    EPOCHS = 60
    BATCH_SIZE = 64
    GPU_NO = 4
    WORKERS = 10
    MULTIPROCESSING = True
    PRETRAINED_WEIGHTS = None # offer the pretrained weights path here
else:
    INPUT_EXAMPLE = '/Users/candrikos/python_workspace/isic_challenge_2018/task_2/ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg'
    OUTPUT_EXAMPLE = "/Users/candrikos/python_workspace/isic_challenge_2018/task_2/ISIC2018_Task2_Training_GroundTruth_v3/ISIC_0000000_attribute_globules.png"
    EPOCHS = 4
    BATCH_SIZE = 32
    GPU_NO = 1
    WORKERS = 1
    MULTIPROCESSING = False
    PRETRAINED_WEIGHTS = None # offer the pretrained weights path here

# this is the accuracy you need to rich ... early stopping will be applied ... required for transfer learning
THRESHOLD_ACCURACY = 0.6
MIN_EPOCHS = 1
EARLY_STOPPING = True