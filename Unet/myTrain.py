'''
    Training UNET model to get the corresponding results
'''
# use the tf embeded keras
#import sys
#import tensorflow.keras as K
#sys.modules["keras"] = K

from Dataloader import createGens
from myModels import Unet as Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from utils import meetExpectations

#import tensorflow as tf
#run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)

from config import EPOCHS, BATCH_SIZE, GPU_NO, WORKERS, MULTIPROCESSING, PRETRAINED_WEIGHTS, THRESHOLD_ACCURACY, MIN_EPOCHS, EARLY_STOPPING

pretrained_weights=PRETRAINED_WEIGHTS    

if __name__ == "__main__":
    model = Model(input_size=(256, 256, 1))

    parallel_model=None
    if GPU_NO > 1:
        parallel_model = multi_gpu_model(model, gpus=GPU_NO)
    final_model = parallel_model or model
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    final_model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    train_gen, val_gen = createGens(batch_size=BATCH_SIZE, split=0.2)


    if EARLY_STOPPING:
        callbacks = [meetExpectations(THRESHOLD_ACCURACY, MIN_EPOCHS, monitor="val_acc" , verbose=1)]
    else:
        callbacks = []

    final_model.fit_generator(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=50,
        validation_data = val_gen,
        validation_steps = 4,
        use_multiprocessing=MULTIPROCESSING,
        workers=WORKERS,
        max_queue_size=WORKERS,
        callbacks = callbacks
    )

    model.save_weights("../weights/model_weights_thresholded.h5")
