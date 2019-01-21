'''
    Training UNET model to get the corresponding results
'''

from sequenceLoader import createGens
from myModels import Unet as Model
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

GPU_NO = 1
pretrained_weights=None
    
if __name__ == "__main__":
    model = Model(input_size=(256, 256, 1))

    parallel_model=None
    if GPU_NO > 1:
        parallel_model = multi_gpu_model(model, gpus=GPU_NO)
    final_model = parallel_model or model
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    final_model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    train_gen, val_gen = createGens()

    model.fit_generator(
        train_gen,
        epochs=2,
        steps_per_epoch=4,
        validation_data = val_gen,
        validation_steps = 2,
        max_queue_size=10
    )

    model.save_weights("model_weights.h5")
