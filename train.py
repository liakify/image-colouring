'''
Trains a model and saves it into the models folder.
Also saves the train test ID split into train.npy and test.npy so that the test script
    knows which IDs to test on.
'''

import data
import models
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ids = data.getImageIds(0.01)
trainIds, testIds = train_test_split(ids, test_size=0.2, random_state=42)
print("Len of trainIds:", len(trainIds))
print("Len of testIds:", len(testIds))

np.save("npy/CIE94_train_1_20_32_10000", trainIds)
np.save("npy/CIE94_test_1_20_32_10000", testIds)

X_train, Y_train = data.loadImageData(trainIds)

# Model specific code

# MSE model
Y_train /= 128
model = models.getCIE94Model()

model.fit(x=X_train, 
    y=Y_train,
    batch_size=32,
    epochs=10000)

model.save("models/CIEmodel_1_20_32_10000")


# Classification model
'''
bins = np.load("npy/pts_in_hull.npy")
model = models.getClassificationModel()
epochs = 100
with tf.device("cpu:0"):
    for i in range(epochs):
        for id in trainIds:
            X_train, Y_train = data.loadImageData([id])
            Y_train = data.batchQuantize(Y_train, bins)

            model.fit(x=X_train, 
                y=Y_train,
                batch_size=1,
                epochs=1)

model.save("models/ClassificationModel")
'''
