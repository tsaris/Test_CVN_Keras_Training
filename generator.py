from __future__ import print_function

import numpy as np
from keras.utils import np_utils


def produce_seq(batch_size, data, sample_ids):
    while True:
        sample_count = len(sample_ids)
        batches = int(sample_count/batch_size)
        left_samples = sample_count%batch_size
        if left_samples:
            batches = batches +1
        for ids in range(0, batches):
            if ids == batches-1:
                batch_ids = sample_ids[ids*batch_size:]
            else:
                batch_ids = sample_ids[ids*batch_size:ids*batch_size+batch_size]
            batch_ids = sorted(batch_ids)

            X = data['data'][batch_ids]
            Y = data['label'][batch_ids]
            X1,X2 = X[:, [0]], X[:, [1]]

            X1 = X1.reshape(X1.shape[0], 1, 100, 80).astype('float32')/255
            X2 = X2.reshape(X2.shape[0], 1, 100, 80).astype('float32')/255
            Y = np_utils.to_categorical(Y,5)

            yield {'input1':np.array(X1), 'input2':np.array(X2)} , { 'out':np.array(Y) }
            
