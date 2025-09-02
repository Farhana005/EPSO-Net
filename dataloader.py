import os
import cv2
import numpy as np
import nibabel as nib
from tensorflow import keras

IMG_SIZE = 128
TRAIN_DATASET_PATH = "E:/datasets/BraTS2021_Training_Data"


import os
import cv2
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras

class DataGenerator1(keras.utils.Sequence):
    def __init__(self, list_IDs, 
                 img_size=128, 
                 num_classes=4, 
                 batch_size=1, 
                 n_channels=3, 
                 shuffle=True, 
                 augment=True, 
                 data_path="E:/datasets/BraTS2021_Training_Data/"):
        self.dim = (img_size, img_size, img_size)
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augment
        self.num_classes = num_classes
        self.data_path = data_path
        self.on_epoch_end()
        # Augmenters (for images and masks, with identical params)
        if augment:
            self.image_datagen = keras.preprocessing.image.ImageDataGenerator(

                rotation_range=35,
                width_shift_range=0.45,
                height_shift_range=0.45,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
            self.mask_datagen = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=35,
                width_shift_range=0.45,
                height_shift_range=0.45,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Batch_ids = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(Batch_ids)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Y = np.zeros((self.batch_size, *self.dim, self.num_classes), dtype=np.float32)
        for c, case_id in enumerate(Batch_ids):
            case_path = os.path.join(self.data_path, case_id)
            flair = nib.load(os.path.join(case_path, f'{case_id}_flair.nii.gz')).get_fdata()
            ce    = nib.load(os.path.join(case_path, f'{case_id}_t1ce.nii.gz')).get_fdata()
            t2    = nib.load(os.path.join(case_path, f'{case_id}_t2.nii.gz')).get_fdata()
            seg   = nib.load(os.path.join(case_path, f'{case_id}_seg.nii.gz')).get_fdata()
            seg[seg == 4] = 3
            # Prepare 3-channel image and 1-hot mask
            x_vol = np.zeros((*self.dim, 3), dtype=np.float32)
            y_vol = np.zeros((*self.dim, self.num_classes), dtype=np.float32)
            for i in range(self.dim[2]):
                flair_slice = cv2.resize(flair[:, :, i], (self.dim[0], self.dim[1]))
                ce_slice    = cv2.resize(ce[:, :, i],    (self.dim[0], self.dim[1]))
                t2_slice    = cv2.resize(t2[:, :, i],    (self.dim[0], self.dim[1]))
                seg_slice   = cv2.resize(seg[:, :, i],   (self.dim[0], self.dim[1]), interpolation=cv2.INTER_NEAREST)
                x_vol[:, :, i, 0] = flair_slice
                x_vol[:, :, i, 1] = ce_slice
                x_vol[:, :, i, 2] = t2_slice
                one_hot_mask = tf.one_hot(seg_slice.astype(np.uint8), depth=self.num_classes)
                y_vol[:, :, i] = one_hot_mask.numpy()
            # Normalize
            x_vol = x_vol / (np.max(x_vol) + 1e-6)
            # Augmentation (slice by slice)
            if self.augment:
                aug_x_vol = np.zeros_like(x_vol)
                aug_y_vol = np.zeros_like(y_vol)
                for i in range(self.dim[2]):
                    img_slice  = x_vol[:, :, i, :]
                    mask_slice = y_vol[:, :, i, :]
                    seed = np.random.randint(1e6)
                    aug_img  = self.image_datagen.random_transform(img_slice, seed=seed)
                    aug_mask = self.mask_datagen.random_transform(mask_slice, seed=seed)
                    aug_y_vol[:, :, i] = np.round(aug_mask)
                    aug_x_vol[:, :, i] = aug_img
                x_vol = aug_x_vol
                y_vol = aug_y_vol
            X[c] = x_vol
            Y[c] = y_vol
        return X, Y


class DataGenerator2(keras.utils.Sequence):
    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=2, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Batch_ids = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(Batch_ids)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.dim, 3), dtype=np.float32)
        def convert_to_wt_tc_et(label):
            wt = np.logical_or.reduce([label == 1, label == 2, label == 4])
            tc = np.logical_or(label == 1, label == 4)
            et = (label == 4)
            return np.stack([wt, tc, et], axis=-1).astype(np.float32)
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)
            flair = nib.load(os.path.join(case_path, f'{i}_flair.nii.gz')).get_fdata()
            ce = nib.load(os.path.join(case_path, f'{i}_t1ce.nii.gz')).get_fdata()
            seg = nib.load(os.path.join(case_path, f'{i}_seg.nii.gz')).get_fdata()
            flair_resized = np.zeros((IMG_SIZE, IMG_SIZE, self.dim[2]))
            ce_resized = np.zeros((IMG_SIZE, IMG_SIZE, self.dim[2]))
            seg_resized = np.zeros((IMG_SIZE, IMG_SIZE, self.dim[2]))
            for j in range(self.dim[2]):
                flair_resized[:, :, j] = cv2.resize(flair[:, :, j], (IMG_SIZE, IMG_SIZE))
                ce_resized[:, :, j] = cv2.resize(ce[:, :, j], (IMG_SIZE, IMG_SIZE))
                seg_resized[:, :, j] = cv2.resize(seg[:, :, j], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            flair_resized /= np.max(flair_resized) + 1e-6
            ce_resized /= np.max(ce_resized) + 1e-6
            X[c, ..., 0] = flair_resized
            X[c, ..., 1] = ce_resized
            y[c] = convert_to_wt_tc_et(seg_resized)
        return X, y
