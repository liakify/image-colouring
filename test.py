'''
Loads a model and produces images based on the test set specified.
'''

import data
import numpy as np
from tensorflow import keras

model = keras.models.load_model("models/MSEmodel")
testIds = np.load("npy/test.npy")
X_test, Y_test = data.loadImageData(testIds)

model.evaluate(X_test, Y_test, batch_size=1)
predictions = model.predict(X_test)

data.generateImages(X_test, predictions * 127, testIds)
