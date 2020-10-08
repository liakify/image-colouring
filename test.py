'''
Loads a model and produces images based on the test set specified.
'''

import data
import numpy as np
from tensorflow import keras

testIds = np.load("npy/test.npy")
X_test, Y_test = data.loadImageData(testIds)

# Model specific code
'''
# MSE model
model = keras.models.load_model("models/MSEmodel")

model.evaluate(X_test, Y_test, batch_size=1)

predictions = model.predict(X_test)
data.generateImages(X_test, predictions * 127, testIds)
'''

# Classification model
model = keras.models.load_model("models/ClassificationModel")
bins = np.load("npy/pts_in_hull.npy")

Y_test = data.batchQuantize(Y_test, bins)
model.evaluate(X_test, Y_test, batch_size=1)

predictions = model.predict(X_test)
AB = data.batchUnquantize(predictions, bins)
data.generateImages(X_test, AB, testIds)
