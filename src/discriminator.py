from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np

class Discriminator:

    def __init__(self, img_shape):
        self.img_shape = img_shape
        print('Discriminator Image shape:', self.img_shape)

    def build_discriminator(self, input_size, dropout=0.25, momentum=0.8):
        ## Dropout is advised to be between 0.4 and 0.7
        model = Sequential()
        ## reduccion del tamano con strides=2
        model.add(Conv2D(input_size, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        ## Recordar padding='same'
        model.add(Conv2D(input_size * 2, kernel_size=3, strides=2, padding="same"))
        # model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        model.add(Conv2D(input_size * 3, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=momentum))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(dropout))

        ## Add 4 layers with default params
        self.addNLayers(4, model, size=256)

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        print('\nDISCRIMINATOR SUMMARY:')
        model.summary()
        ## Store summary to textfile
        ## Pass the file handle in as a lambda function to make it callable
        with open('output/discriminator__summary.txt','w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        self.Model = Model(img, validity)
        return self.Model
    
    def addNLayers(self, n, model, size=256, kernel_s=3, strides=1, momentum=0.8, alpha=0.2, dropout=0.2):
        for i in range(n):
            model.add(Conv2D(size, kernel_size=kernel_s, strides=strides, padding="same"))
            model.add(BatchNormalization(momentum=momentum))
            model.add(LeakyReLU(alpha=alpha))
            model.add(Dropout(dropout))