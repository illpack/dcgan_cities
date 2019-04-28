DCGANs
===

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