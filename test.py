'''
Loads a model and produces images based on the test set specified.
'''

import data
import numpy as np
import tensorflow as tf
from tensorflow import keras

testIds = np.load("npy/test.npy")
X_test, Y_test = data.loadImageData(testIds)

# Same here
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Model specific code
# MSE model
model = keras.models.load_model("models/MSEmodel")

model.evaluate(X_test, Y_test, batch_size=1)

predictions = model.predict(X_test)
data.generateImages(X_test, predictions * 127, testIds)

# Classification model
'''
model = keras.models.load_model("models/ClassificationModel")
bins = np.load("npy/pts_in_hull.npy")

Y_test = data.batchQuantize(Y_test, bins)
model.evaluate(X_test, Y_test, batch_size=1)

for id in testIds:
    X_test, Y_test = data.loadImageData([id])
    predictions = model.predict(X_test)
    AB = data.batchUnquantize(predictions, bins)
    data.generateImages(X_test, AB, [id])
'''