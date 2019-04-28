from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np

class Generator:

    def __init__(self, img_shape, input_size, latent_dim, channels):
        self.img_shape = img_shape
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.channels = channels
        print('Generator Image shape:', self.img_shape)
    
    
    def build_generator(self, momentum=0.8):
    
        ##Dropout of between 0.3 and 0.5 at the first layer prevents overfitting.
    
        model = Sequential()

        model.add(Dense(self.input_size * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, self.input_size)))
        model.add(UpSampling2D())

        model.add(Conv2D(self.input_size, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        ## TODO: Dropout?

        model.add(Conv2D(self.input_size, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation("relu"))
        model.add(UpSampling2D())

        model.add(Conv2D(self.input_size, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation("relu"))
        model.add(UpSampling2D())

        model.add(Conv2D(self.input_size, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        ## Added to get the right output
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        print('\nGENERATOR SUMMARY:')
        model.summary()
        ## Store summary to textfile
        ## Pass the file handle in as a lambda function to make it callable
        with open('output/generator__summary.txt','w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        self.Model = Model(noise, img)
        return self.Model