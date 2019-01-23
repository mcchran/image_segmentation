PROJECT_ROOT = "." #TODO: add the project root directory here ...

DESTINATION = "hpc"

if DESTINATION=="hpc":
    IMAGE_PATHS = '../data/ChinaSet_AllFiles/CXR_png'
    MASK_PATHS = '../data/mask'
    EPOCHS = 60
    BATCH_SIZE = 64
    GPU_NO = 4
    WORKERS = 10
    MULTIPROCESSING = True
    PRETRAINED_WEIGHTS = None # offer the pretrained weights path here
else:
    IMAGE_PATHS = '../data/train_data/ChinaSet_AllFiles/CXR_png'
    MASK_PATHS = "../data/train_data/mask"
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
