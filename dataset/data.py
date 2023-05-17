import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST, CelebA
from torchvision import transforms
from torch.utils.data import Dataset
import os
import torchvision
from PIL import  Image

class Celeba(Dataset):
    def __init__(self, path, transform=None):
        super(Celeba, self).__init__()
        self.files = []
        self.transform = transform
        for root, dirs, files in os.walk(path):
            self.files += [
                os.path.join(root, _)
                for _ in files if os.path.splitext(_)[-1] == '.jpg'
            ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path = self.files[item]
        img = Image.open(path)
        return self.transform(img.convert('RGB'))


class CELEBADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, image_size=128, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(256),
            transforms.ToTensor()
        ])

    def setup(self, stage: str):
        self.celeba_all = Celeba(self.data_dir, transform=self.transform)
        l = int(len(self.celeba_all) * 0.10)
        lrest = len(self.celeba_all) - l
        self.celeba_train, self.celeba_val = random_split(self.celeba_all, [lrest, l])

    def train_dataloader(self):
        return DataLoader(self.celeba_train, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=4)

    def val_dataloader(self):
        return DataLoader(self.celeba_val, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=4)

    def test_dataloader(self):
        return DataLoader(self.celeba_val, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=4)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32, image_size=256, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        print(data_dir)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.PILToTensor()
        ])

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)
        self.mnist_predict = MNIST(self.data_dir, train=False, download=True, transform=self.transform)
        mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass