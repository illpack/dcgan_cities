from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import datetime, time
import sys
import logging
import numpy as np
import scipy.misc # export images

## Custom classes
from discriminator import Discriminator
from generator import Generator

class DCGAN():
    
    def __init__(self, rows, cols, channels, data, input_size=128):
        # Input shape
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        ## Encoded dimensions (Autoencoder)
        self.latent_dim = 100
        ## Loaded data 
        self.data = data 
        self.input_size = input_size 
        self.setupLogs()
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        dc = Discriminator(self.img_shape)
        self.discriminator = dc.build_discriminator(input_size=32)
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        gn = Generator(self.img_shape, self.input_size, self.latent_dim, self.channels)
        self.generator = gn.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def train(self, epochs, batch_size=128, save_interval=50):
    
        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        X_train = np.array(self.data)

        # Rescale -1 to 1
        ## Pablo dice que el cero sea significativo
        X_train = (X_train) / 255 - 0.5
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            print('\nEPOCH:', str(epoch), '.......................')
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            self.saveImg(gen_imgs[0], str(epoch))
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            epoch_info = "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss)
            logging.info(epoch_info)
            print (epoch_info)

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        # fig.savefig("images/mnist_%d.png" % epoch)
        timestamp = str(time.time()).split('.')[0]
        fig.savefig("output/gen__{0}_epoch_{1}.png".format(timestamp, str(epoch)))
        print('Image has been saved')
        plt.close()

    def saveImg(self, img, name=''):
        timestamp = str(time.time()).split('.')[0]
        scipy.misc.toimage(img).save('output/process/{0}_{1}.jpg'.format(timestamp, name))

    def setupLogs(self):
        fmtstr = "%(asctime)s: %(levelname)s: %(funcName)s: Line:%(lineno)d %(message)s"
        dtstr  = "%m/%d/%Y %H:%M:%S %p"

        logging.basicConfig(filename = 'output.log'
                            ,level=logging.DEBUG
                            ,filemode='w'
                            ,format=fmtstr
                            ,datefmt=dtstr
                            )


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=1)