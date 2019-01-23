    #
    #	Utilities file 
    #

from keras.callbacks import EarlyStopping

class meetExpectations(EarlyStopping):
    '''
        Applies an early stopping when a metric threshold is reached
        TODO: now the metric applies only to the val_acc but this must be defined though the parameter "monitor"
    '''
    def __init__ (self, threshold, min_epochs, **kwargs):
        super(meetExpectations, self).__init__(**kwargs)
        self.threshold = threshold
        self.min_epochs = min_epochs

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor) # self_monitor is the way to get the monitored metric ... 
        
        if current is None:
            super.warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        
        if (epoch >= self.min_epochs) & (current >= self.threshold):
            self.stopped_epoch = epoch
            print("The model now stops because of reaching the val_acc threshold, in epoch: ", epoch)
            self.model.stop_training = True

class adaptLearningRate(EarlyStopping):
    '''
        Applies updates learning rate to adapt to steep slopes.
    '''
    #TODO: to be done
    pass
