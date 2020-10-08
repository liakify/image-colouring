'''
Trains a model and saves it into the models folder.
Also saves the train test ID split into train.npy and test.npy so that the test script
    knows which IDs to test on.
'''

import data
import models
import numpy as np
from sklearn.model_selection import train_test_split

ids = data.getImageIds(0.01)
trainIds, testIds = train_test_split(ids, test_size=0.2, random_state=42)

np.save("npy/train", trainIds)
np.save("npy/test", testIds)

X_train, Y_train = data.loadImageData(trainIds)

# Model specific code
'''
# MSE model
Y_train /= 128
model = models.getMSEModel()

model.fit(x=X_train, 
    y=Y_train,
    batch_size=1,
    epochs=100)

model.save("models/MSEmodel")
'''
