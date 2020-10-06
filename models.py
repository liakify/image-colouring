'''
Contains the models used in the project.
'''

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras.models import Sequential


'''
Returns the loss between the predicted colours and the ground truth colours of a specific training instance. The calculation is based on the
CIE94 colour distance metric that is an improvement over pure RMSE on the L*a*b* vectors.

NOTE: 
    1. This function is currently hardcoded to be specifically for an image resolution of 384 * 384, and for training batch size 1.
'''
def cie94(y_true, y_pred):
    alpha = []
    loss = 0.0
    for x in range(0, 384*384):
        alpha.append(x*2)
    beta = []
    for y in range(1, 384*384 + 1):
        beta.append(y*2 - 1)
        
    batch_size = 1 # Batch size is hardcoded, reading y_true.shape[0] is problematic on different environments.
    normalizing_constant = batch_size * 384 * 384 # Height and Width are hardcoded, reading y_true.shape[1] and y_true.shape[2] is problematic on different environments.
    for i in range(0, batch_size):
    
        trueAlpha = tf.keras.backend.gather(tf.keras.backend.flatten(y_true[i]), alpha)
        predAlpha = tf.keras.backend.gather(tf.keras.backend.flatten(y_pred[i]), alpha)
    
        trueBeta = tf.keras.backend.gather(tf.keras.backend.flatten(y_true[i]), beta)
        predBeta = tf.keras.backend.gather(tf.keras.backend.flatten(y_pred[i]), beta)
    
        trueAlphaSqr = tf.keras.backend.square(trueAlpha)
        trueBetaSqr = tf.keras.backend.square(trueBeta)
        predAlphaSqr = tf.keras.backend.square(predAlpha)
        predBetaSqr = tf.keras.backend.square(predBeta)
    
        C1 = tf.keras.backend.sqrt(trueAlphaSqr + trueBetaSqr)
        C2 = tf.keras.backend.sqrt(predAlphaSqr + predBetaSqr)
        delta_C = C1 - C2
        delta_Csqr = tf.keras.backend.square(delta_C)
        delta_a = trueAlpha - predAlpha
        delta_b = trueBeta - predBeta
        delta_H_square = tf.keras.backend.square(delta_a) + tf.keras.backend.square(delta_b) - delta_Csqr
        K_1 = 0.045 
        K_2 = 0.015
        
        loss += tf.keras.backend.sum(tf.keras.backend.sqrt(delta_Csqr / tf.keras.backend.square(1.0 + K_1 * C1) + delta_H_square / tf.keras.backend.square(1.0 + K_2 * C1)), axis=0)
    return loss / normalizing_constant

'''
Returns the model object for image colouring via MSE loss.
'''
def getMSEModel():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

    model.compile(optimizer='rmsprop',loss='mse')

    return model

'''
Returns the model object for image colouring via CIE94 Colour Distance loss.
'''
def getCIE94Model():
    model = Sequential()
    model.add(InputLayer(input_shape=(None, None, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

    model.compile(optimizer='rmsprop',loss=cie94)    

