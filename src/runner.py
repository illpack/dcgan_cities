import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os, sys, argparse
from image_loader import ImageLoader
from dcgan import DCGAN

folder = 'C:/Users/lhrfxg/workspace/1902_fastai/datasets/codalab/'

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datalen', type=int, nargs='?', help='Number of training samples. Deafult: All')
    parser.add_argument('--folder', type=str, default=folder, nargs='?', help='folder to images')
    parser.add_argument('--batch_size', type=int, nargs='?', default=32, help='Batch size, default 32')
    parser.add_argument('--epochs', type=int, nargs='?', default=100, help='Epochs, default 100')
    parser.add_argument('--save_interval', type=int, nargs='?', default=10, help='Save interval, default 10')

    main(**vars(parser.parse_args()))

def main(**kwargs):
    from types import SimpleNamespace
    _ = SimpleNamespace(**kwargs)

    loader = ImageLoader(_.folder)
    data = loader.setup(datalen=_.datalen)
    dcgan = DCGAN(loader.shape_x, loader.shape_y, loader.channels, data)
    dcgan.train(epochs=_.epochs, batch_size=_.batch_size, save_interval=50)

if __name__ == '__main__':
    run()