DCGANs
===

_A project by @illpack and @k9martin_

Motivation
---

This project aims to address Urban Networks Design as a [wicked problem](https://en.wikipedia.org/wiki/Wicked_problem), _a problem whose social complexity means that it has no determinable stopping point_. Can Machine and Deep Learning techniques introduce significant changes in the way we design? Will AI overcome parametricism as the contemporary paradigm driving advanced architectural, structural and urban design?

The project can be divided in three sections: Scrapping, Feature extraction, Generation.

### Scraping

We have used urban street networks as a proof of concept. To this end, we have generated a 6,6K+ images dataset from Open Street Maps, taking background-figure snapshots of urban networks. 

### Feature extraction

We have used deep convolutional autoencoders to reduce dimensionality and extract a vector of most relevant features. Then we have clustered this lower dimensional output and observed the groups: while certain composition bias can be ascertained, the  

### Generation

Usage via command line interface:

```
usage: runner.py [-h] [--datalen [DATALEN]] [--folder [FOLDER]]
                 [--batch_size [BATCH_SIZE]] [--epochs [EPOCHS]]
                 [--save_interval [SAVE_INTERVAL]]

optional arguments:
  -h, --help                    show this help message and exit
  --datalen [DATALEN]           Number of training samples. Deafult: All
  --folder [FOLDER]             folder to images
  --batch_size [BATCH_SIZE]     Batch size, default 32
  --epochs [EPOCHS]             Epochs, default 100
  --save_interval [SAVE_INTERVAL]
                        Save interval, default 10
```

### Resources

#### [Build a GAN with Keras](https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0)

*   [__Strided Convolution__](https://www.coursera.org/lecture/convolutional-neural-networks/strided-convolutions-wfUhx)
    This means convolving a `NxN` matrix with a `fxf` filter applying a padding of `p` and a stride of `s`. The stride equals the number of steps that the filter will jump per iteration. So a stride of 2 would reduce the input to `0.5*Nx0.5*N`. Generally: `output = (N + 2*p - f) / s + 1`


*   [__Max-pooling__](https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks)
