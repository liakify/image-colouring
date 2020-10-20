'''
Loads a model and produces images based on the test set specified.
'''

import data
import numpy as np
from tensorflow import keras

index = "502_160_40_200"
trainIds = np.load("npy/train_{}.npy".format(index))
testIds = np.load("npy/test_{}.npy".format(index))


# Model specific code

# MSE model
model = keras.models.load_model("models/CIEmodel_502_160_40_200", compile=False)
'''
X, Y = data.loadImageData(trainIds)
print("Train error")
model.evaluate(X, Y, batch_size=32)

X, Y = data.loadImageData(testIds)
print("Test error")
model.evaluate(X, Y, batch_size=32)
'''
X_test, Y_test = data.loadImageData(testIds)
predictions = model.predict(X_test)
data.generateImages(X_test, predictions * 127, testIds)

'''
# Classification model
model = keras.models.load_model("models/ClassificationModel_50_20_4_100")
bins = np.load("npy/pts_in_hull.npy")

for ids in [trainIds, testIds]:
    print(len(ids))
    loss = 0.0
    for id in ids:
        X, Y = data.loadImageData([id])
        Y = data.batchQuantize(Y, bins)
        loss += model.evaluate(X, Y, verbose=0)
    print("Loss: {}".format(loss / len(ids)))
''' 
'''
for id in testIds:    
    X_test, Y_test = data.loadImageData([id])
    predictions = model.predict(X_test)
    AB = data.batchUnquantize(predictions, bins)
    data.generateImages(X_test, AB, [id])
'''
