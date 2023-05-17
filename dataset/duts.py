import torch
import torch.nn as nn
import torchvision

import  os
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
from typing import Union
import pytorch_lightning as pl


def pad_resize_image(inp_img, out_img=None, target_size=None):
    h, w, c = inp_img.shape
    size = max(h, w)
    padding_h = (size - h) // 2
    padding_w = (size - w) // 2
    if out_img is None:
        temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h,
                                    left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if target_size is not None:
            temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return temp_x
    else:
        temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        temp_y = cv2.copyMakeBorder(out_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if target_size is not None:
            temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
            temp_y = cv2.resize(temp_y, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return temp_x, temp_y


def random_crop_flip(inp_img, out_img):
    h, w = out_img.shape
    rand_h = np.random.randint(h/8)
    rand_w = np.random.randint(w/8)
    offset_h = 0 if rand_h == 0 else np.random.randint(rand_h)
    offset_w = 0 if rand_w == 0 else np.random.randint(rand_w)
    p0, p1, p2, p3 = offset_h, h+offset_h-rand_h, offset_w, w+offset_w-rand_w
    rand_flip = np.random.randint(10)
    if rand_flip >= 5:
        inp_img = inp_img[::, ::-1, ::]
        out_img = out_img[::, ::-1]
    return inp_img[p0:p1, p2:p3], out_img[p0:p1, p2:p3]


def random_rotate(inp_img, out_img, max_angle=25):
    angle = np.random.randint(-max_angle, max_angle)
    h, w = out_img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(inp_img, M, (new_w, new_h)), cv2.warpAffine(out_img, M, (new_w, new_h))


def random_rotate_lossy(inp_img, out_img, max_angle=25):
    angle = np.random.randint(-max_angle, max_angle)
    h, w = out_img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(inp_img, M, (w, h)), cv2.warpAffine(out_img, M, (w, h))


def random_brightness(inp_img):
    contrast = np.random.rand(1) + 0.5
    light = np.random.randint(-20, 20)
    inp_img = contrast * inp_img + light
    return np.clip(inp_img, 0, 255)


class SODLoader(Dataset):
    def __init__(self,
                 duts_base: Union[os.PathLike, str],
                 mode='train',
                 augment_data=False,
                 target_size=256,
                 repeat=1):
        assert mode in ['train', 'test']
        self.image_path = os.path.join(duts_base, 'DUTS-TR-Image' if mode == 'train' else 'DUTS-TE-Image')
        self.mask_path = os.path.join(duts_base, 'DUTS-TR-Mask' if mode == 'train' else 'DUTS-TE-Mask')

        self.augment_data = augment_data
        self.target_size = target_size
        self.repeat = repeat
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.image_path = list(sorted(glob.glob(self.image_path + '/*')))*self.repeat
        self.mask_path = list(sorted(glob.glob(self.mask_path + '/*')))*self.repeat

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        inp_img = cv2.imread(self.image_path[idx])
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
        inp_img = inp_img.astype('float32')

        mask_img = cv2.imread(self.mask_path[idx], 0)
        mask_img = mask_img.astype('float32')
        mask_img /= np.max(mask_img)

        if self.augment_data:
            inp_img, mask_img = random_crop_flip(inp_img, mask_img)
            inp_img, mask_img = random_rotate(inp_img, mask_img)
            inp_img = random_brightness(inp_img)

        # Pad images to target size
        inp_img, mask_img = pad_resize_image(inp_img, mask_img, self.target_size)
        inp_img /= 255.0
        inp_img = np.transpose(inp_img, axes=(2, 0, 1))
        inp_img = torch.from_numpy(inp_img).float()
        # inp_img = self.normalize(inp_img)

        mask_img = np.expand_dims(mask_img, axis=0)

        return inp_img, torch.from_numpy(mask_img).float()


class DutsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, image_size=288, num_workers=8, repeat=2, augment=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.num_workers = num_workers
        self.repeat = repeat


    def setup(self, stage: str):
        self.duts_train = SODLoader(
            os.path.join(self.data_dir, 'DUTS-TR'),
            'train',
            augment_data=self.augment,
            target_size=self.image_size,
            repeat=self.repeat
        )

        self.duts_test = SODLoader(
            os.path.join(self.data_dir, 'DUTS-TE'),
            'test',
            augment_data=False,
            target_size=self.image_size,
            repeat=self.repeat
        )


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.duts_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.duts_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.duts_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.duts_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass