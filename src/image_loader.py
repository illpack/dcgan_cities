import numpy as np
import os, sys
from PIL import Image

class ImageLoader():

    def __init__(self, folder, shape_x=256, shape_y=256, channels=3):
        self.folder = folder
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.channels = channels
        self.X = []

    def setup(self, extension='.jpg', datalen=None):
        try:
            assert os.path.isdir(self.folder)
            print('Loading images...')
            files = list(filter(lambda x: x.endswith(extension), os.listdir(self.folder)))
            if not datalen:
                datalen = len(os.listdir(self.folder))
            for idx, img in enumerate(files):
                image_path = os.path.join(self.folder, img)
                im = Image.open(image_path)
                im = im.resize((self.shape_x, self.shape_y), Image.ANTIALIAS)
                if image_path.endswith(extension):
                    self.X.append(np.array(im))
                    if (idx % 50 == 0): 
                        print(round(100*idx/datalen, 1), sep='...')
                    if (datalen and idx>datalen):
                        print(np.array(self.X).shape)
                        return self.X
            print(np.array(self.X).shape)
            return self.X
        
        except AssertionError:
            sys.exit('Folder does not exist')


