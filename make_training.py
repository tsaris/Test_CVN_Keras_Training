############################################
# First load all the modules in the begining
############################################
from __future__ import print_function
import os

# To save dictionaries
import pickle

# Keras utilities
import keras
from keras.models import model_from_json, load_model
from keras.optimizers import SGD
from keras.utils import np_utils

import h5py
import numpy as np

# Local models
import models
import generator
from multi_gpu import make_parallel

dataset = "train.h5"
hf = h5py.File(dataset, 'r')
n1 = hf.get('data')
total_count = n1.shape[0]
hf.close()

def train_model(model, dataset, validation_ratio=0.2, batch_size=64):
    with h5py.File(dataset, "r") as data:

        total_ids = range(0, total_count)
        total_ids = np.random.permutation(total_ids)
        train_total_ids = total_ids[0:int((1-validation_ratio)*total_count)]
        test_total_ids = total_ids[int((1-validation_ratio)*total_count):]

        training_sequence_generator = generator.produce_seq(batch_size=batch_size, 
                                                            data=data, sample_ids=train_total_ids)
        validation_sequence_generator = generator.produce_seq(batch_size=batch_size, 
                                                              data=data, sample_ids=test_total_ids)
        
        history = model.fit_generator(generator=training_sequence_generator,
                                      validation_data=validation_sequence_generator,
                                      samples_per_epoch=len(train_total_ids),
                                      nb_val_samples=len(test_total_ids),
                                      nb_epoch=1,
                                      max_q_size=1,
                                      verbose=1,
                                      class_weight=None,
                                      nb_worker=1)

        directory = 'logs/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the history/dictonary to plot it later
        with open(directory + 'history.pickle', 'wb') as handle:
            pickle.dump(history.history, handle, protocol=2)
        print("The training/testing logs saved")

        # serialize model to JSON
        model_json = model.to_json()
        with open(directory + "model.json", "w") as json_file:
            json_file.write(model_json)
        print("The arch saved")

        #serialize weights to HDF5
        model.save_weights(directory + "model_weights.h5")
        model.save(directory + 'model_4recover.h5')
        print("The weights and model saved")
                                      

model = models.CVN(5)
#model = make_parallel(model, 2)
learning_rate = 0.02
decay_rate = 0.1
momentum = 0.9
opt = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc','top_k_categorical_accuracy'])

train_model(model, dataset)
