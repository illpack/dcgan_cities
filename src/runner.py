import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os, sys
from image_loader import ImageLoader
from dcgan import DCGAN

def main():
    loader = ImageLoader('C:/Users/lhrfxg/workspace/1902_fastai/datasets/codalab/')
    data = loader.setup(datalen=200)
    dcgan = DCGAN(loader.shape_x, loader.shape_y, loader.channels, data)
    dcgan.train(epochs=100, batch_size=32, save_interval=50)

if __name__ == '__main__':
    main()