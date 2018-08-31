#!/usr/bin/python

from __future__ import print_function

###########################################################
# From LevelDB 2 HDF5                                      #
# The number of hdf5s comes from caffe limitation for hdf5 #
# data layer, since there is no tranformation              #
# they have hardcoded a value of 2147483647                #
# So or our case batch size 16                             #
# 2147483647 / (16*100*80) = 16777                         #
############################################################

# Need for the leveldb
import caffe
import leveldb
from caffe.proto import caffe_pb2

# Need for the hdf5
import h5py

from random import shuffle

# ----------------------------------- #
# ---------- Vars ------------------- #
# ----------------------------------- #

test_caffe_limit = 20000
train_caffe_limit = 80000

test_total_limit = 200000
train_total_limit = 800000

# List of leveldb dirs

ldb_dirs=["flat/1/output50full",
          "flat/2/output50full",
          "flat/3/output50full",
          "flat/4/output50full",
          "flat/5/output50full",
          "flat/6/output50full",
          "flat/7/output50full",
          "flat/8/output50full",
          "flat/9/output50full",
          "flat/10/output50full"]


# ----------------------------------- #
# ------------- Main ---------------- #
# ----------------------------------- #
test_images = []
test_labels = []
train_images = []
train_labels = []
count_test = 0
count_train = 0

for ldb_dir in ldb_dirs:

    print("Start with the test")
    db_path = leveldb.LevelDB(ldb_dir+"/TestLDB_4V_purity50_full")
    datum = caffe_pb2.Datum()
    for key, value in db_path.RangeIter():
       
        # Fill out the lists
        datum.ParseFromString(value)
        if (datum.label == 1): continue # skip if there is a muon
        test_images.append( caffe.io.datum_to_array(datum) )
        test_labels.append( datum.label )

       # Check the counts
        count_test = count_test + 1
        if (count_test%1000==0): print("reading", count_test)

    print("Start with the train")
    db_path = leveldb.LevelDB(ldb_dir+"/TrainLDB_4V_purity50_full")
    datum = caffe_pb2.Datum()
    for key, value in db_path.RangeIter():
       
        # Fill out the lists
        datum.ParseFromString(value)
        if (datum.label == 1): continue # skip if there is a muon
        train_images.append( caffe.io.datum_to_array(datum) )
        train_labels.append( datum.label )

       # Check the counts
        count_train = count_train + 1
        if (count_train%1000==0): print("reading", count_train)


print("Done reading lists")

# Shuffle the list
x = []
for i in range(0, len(test_labels)):
    x.append(i)

shuffle(x)
shuffle(x)
shuffle(x)
shuffle(x)
shuffle(x)

for i in range(0, len(test_labels)):
    tmp_image = test_images[i]
    tmp_label = test_labels[i]
    test_images[i] = test_images[ x[i] ]
    test_labels[i] = test_labels[ x[i] ]
    test_images[ x[i] ] = tmp_image
    test_labels[ x[i] ] = tmp_label

y = []
for i in range(0, len(train_labels)):
    y.append(i)

shuffle(y)
for i in range(0, len(train_labels)):
    tmp_image = train_images[i]
    tmp_label = train_labels[i]
    train_images[i] = train_images[ y[i] ]
    train_labels[i] = train_labels[ y[i] ]
    train_images[ y[i] ] = tmp_image
    train_labels[ y[i] ] = tmp_label

print("Done shuffling")

count_test = 0
tmp_images = []
tmp_labels = []
for i in range(0, len(test_labels)):
    tmp_images.append(test_images[i])
    tmp_labels.append(test_labels[i])
    if (count_test%test_caffe_limit==0):
        hf = h5py.File('flat_nonMu/test.genie.prong_'+str(count_test)+'.h5', 'w')
        hf.create_dataset('data', data=tmp_images)
        hf.create_dataset('label', data=tmp_labels)
        hf.close()
        print("Made an hdf5 file with", count_test)
        tmp_images = []
        tmp_labels = []
    count_test = count_test + 1
    if (count_test%1000==0): print("writing", count_test)
    if (count_test%test_total_limit==0): break


# help with the memory
test_images=[]
test_labels=[]

count_train = 0
tmp_images = []
tmp_labels = []
for i in range(0, len(train_labels)):
    tmp_images.append(train_images[i])
    tmp_labels.append(train_labels[i])
    if (count_train%train_caffe_limit==0):
        hf = h5py.File('flat_nonMu/train.genie.prong_'+str(count_train)+'.h5', 'w')
        hf.create_dataset('data', data=tmp_images)
        hf.create_dataset('label', data=tmp_labels)
        hf.close()
        print("Made an hdf5 file with", count_train)
        tmp_images = []
        tmp_labels = []
    count_train = count_train + 1
    if (count_train%1000==0): print("writing", count_train)
    if (count_train%train_total_limit==0): break
        
    
print("Done Done ...")
