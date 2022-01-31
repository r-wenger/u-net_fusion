import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical ,Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adadelta, Nadam ,Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard

import os
from glob import glob
from pathlib import Path
import shutil
from random import sample, choice
import tifffile as tiff
from skimage.transform import rotate
import imageio
import imgaug as ia
from tensorflow.keras.utils import to_categorical
import imgaug.augmenters as iaa


class DataGenerator(Sequence):
    'Generates data for Keras'
    
    def __init__(self, pair, batch_size=64, shuffle=True, dataAugment=True):
        'Initialization'
        self.pair = pair
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataAugment = dataAugment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.pair) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_augmentation(self, img, mask):
        """
        :param img: ndarray with shape (x_sz, y_sz, num_channels)
        :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
        :return: patch with shape (sz, sz, num_channels)
        """

        patch_img = img.astype(float)
        patch_mask = mask

        random_transformation = np.random.randint(1,6)
        if random_transformation == 1: #rotate 90 degrees
            patch_img = np.rot90(patch_img, 1)
            patch_mask = np.rot90(patch_mask, 1)
        elif random_transformation == 2: #rotate 180 degrees
            patch_img = np.rot90(patch_img, 2)
            patch_mask = np.rot90(patch_mask, 2)
        elif random_transformation == 3: #rotate 270 degrees
            patch_img = np.rot90(patch_img, 3)
            patch_mask = np.rot90(patch_mask, 3)
        elif random_transformation == 4: #flipping up to down
            patch_img = np.flipud(patch_img)
            patch_mask = np.flipud(patch_mask)
        elif random_transformation == 5: #flipping left to right
            patch_img = np.fliplr(patch_img)
            patch_mask = np.fliplr(patch_mask)

        return patch_img, patch_mask

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        batch_imgs = list()
        batch_labels = list()

        # Generate data
        for i in list_IDs_temp:
            # Store sample
            img = tiff.imread(str(self.pair[i][0])).astype(float)
            label = tiff.imread(str(self.pair[i][1])).astype(int)

            _label = np.zeros((label.shape[0], label.shape[1]), dtype=int)

            for b in range(0, label.shape[2]):
                _label += np.where(label[:, :, b] == 1, b+1, 0)

            #TODO : can be change
            _label = np.where(_label == 0, 6, _label)

            label = to_categorical(y=_label)[:, :, 1:]

            if self.dataAugment:
                batch_labels.append(label)
                batch_imgs.append(img)
                img, label = self.data_augmentation(img, label)

            batch_labels.append(label)
            batch_imgs.append(img)
            
        return np.array(batch_imgs), np.array(batch_labels)