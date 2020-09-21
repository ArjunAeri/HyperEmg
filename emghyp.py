import pal
import tensorflow.keras as ks
import numpy as np
import pywt
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, GlobalAveragePooling1D, Dense  
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

"""
Arjun Aeri 2020 

This class is designed to allow for streamlined hyperparameter optimization
and then result visualization via tensorboard for this emg dataset and a search space
of different neural networks. 

hparams: A list of tensorboard.plugins.hparams.HParam values. The keys of these will be used in
         modelg, and the values are the ranges of values to be tested for that hyperparameter.

modelg: A function that returns the model you want to hyperparameter optimize.
        The function should have parameters for a hyperparameter dictionary as well as a 
        tuple for the input shape (this is based off the preprocessing you do). Use these parameters
        to populate your model and return it.

prepropf: Pass a list of functions so that preprocessing of the dataset can be changed as a 
          hyperparameter too! Or pass a list of len(1) with one function
          to just use the same preprocessing function.

metric: training/validation metric to use for scoring in tensorboard 
          
"""

class HyperEmg():
    def __init__(self, hparams, modelg, prepropf, metric='accuracy'):
        self.tdx, self.tdy = pal.get_xy() # Read the numpy files (fast) to load the data

        self.hparams = hparams
        self.modelg = modelg
        self.metric = metric
        
        self.session_num = 0       # What number model we are currently training
        self.prepropf = prepropf

        self.prepropds = {}        # Dictionary with keys for each preprocessing function and values of respective dataset
        self.preproc_datasetg()

        self.grid_search(hparams, {}, len(hparams) - 1) # Start the grid search!

    """
    Generates respective dataset for each preprocessing method.
    self.preprods is a dictionary with keys for each preprocessing function.
    This way datasets are cached and don't need to be generated every time we want to train a model
    with a specific preprocessing method.
    """
    def preproc_datasetg(self):
        for currentfunc in self.prepropf:
            self.prepropds[currentfunc.__name__] = currentfunc(self.tdx)
            # Dict key is function name, value is dataset processed by that function

        print(self.prepropds.keys())


    """
    Creates and trains the model with one given set of hyperparameters and 
    returns performance for that model
    We are using end validation accuracy to judge the performance of the model.
    This could be improved in the future with a separate test data set and other metrics.
    """
    def train_test_model(self, hparams):
        preprop_type = hparams['preprocessing']

        model = self.modelg(hparams, self.prepropds[preprop_type].shape[1:])
        
        model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
        
        res = model.fit(self.prepropds[preprop_type], self.tdy, epochs=45, validation_split=.2, batch_size = 32)

        #_, accuracy = model.evaluate(tdx, tdy) this could be used with separate test datset.
        return max(res.history['val_accuracy']) 
    """
    Calls train_test_model and logs the results to tensorboard 
    """
    def run(self, run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = self.train_test_model(hparams)
            tf.summary.scalar(self.metric, accuracy, step=1)
    """
    Performs a grid search on the all the possible permuations of the hyperparameters
    and logs results to tensorboard (calls run on each possible model)
    Recursive depth first search populating the hpd dictionary greedily. 
    """
    def grid_search(self, hps, hpd, depth): 
        if depth == -1:
            self.session_num += 1
            run_name = "run-%d" % self.session_num
            print('--- Starting trial: %s' % run_name)
            #print({h.name: self.hparams[h] for h in hps})
            self.run('logs/hparam_tuning/' + run_name, hpd)
            
        else:
            chp = hps[depth]
            for val in chp.domain.values:
                hpd[chp.name] = val
                self.grid_search(hps, hpd, depth - 1)

    
                
    
    

    

    



