import os
import random
from matplotlib.transforms import Transform
import torch
from torchvision.transforms.functional import vflip, hflip
from torch.utils.data import Dataset
from torchvision.io import read_image

class ImageDataset(Dataset):
    def __init__(self, data_path='./data', option='Train', transform=None, is_test=False):
        self.option = option
        self.is_test = is_test
        self.sem_dir = os.path.join(data_path, option, 'SEM')
        self.depth_dir = os.path.join(data_path, option, 'Depth')

        self.sem_list = sorted(os.listdir(self.sem_dir))
        self.transform = transform

    def __len__(self):
        return len(self.sem_list)

    def __getitem__(self, idx):
        sem_fname = self.sem_list[idx]
        depth_fname = '_'.join(sem_fname.split('_')[:-1]) + '.png'

        sem_path = os.path.join(self.sem_dir, sem_fname)
        sem_image = read_image(sem_path) / 255.

        if self.option == 'Test' or self.is_test:
            return sem_image

        depth_path = os.path.join(self.depth_dir, depth_fname)
        depth_image = read_image(depth_path) / 255.

        if self.transform:
            if random.random() > 0.5:
                sem_image, depth_image = vflip(sem_image), vflip(depth_image)
            if random.random() > 0.5:
                sem_image, depth_image = hflip(sem_image), hflip(depth_image)

        return sem_image, depth_image

class MultiImageDataset(Dataset):
    def __init__(self, data_path='./data', option='Train', transform=None, is_test=False, n_imgs=4):
        self.option = option
        self.is_test = is_test
        self.sem_dir = os.path.join(data_path, option, 'SEM')
        self.depth_dir = os.path.join(data_path, option, 'Depth')
        self.n_imgs = n_imgs

        self.sem_list = sorted(os.listdir(self.sem_dir))
        self.transform = transform

        assert len(self.sem_list) % n_imgs == 0

    def __len__(self):
        return len(self.sem_list) // 4

    def __getitem__(self, idx):
        sem_fnames = self.sem_list[self.n_imgs*idx:self.n_imgs*(idx+1)]
        depth_fnames = ['_'.join(sem_fname.split('_')[:-1]) + '.png' for sem_fname in sem_fnames]
        depth_fnames = list(set(depth_fnames))
        assert len(depth_fnames) == 1

        depth_fname = depth_fnames[0]

        sem_imgs = []
        for sem_fname in sem_fnames:
            sem_path = os.path.join(self.sem_dir, sem_fname)
            sem_image = read_image(sem_path) / 255.
            sem_imgs.append(sem_image)
        sem_imgs = torch.stack(sem_imgs)

        if self.option == 'Test' or self.is_test:
            return sem_imgs

        depth_path = os.path.join(self.depth_dir, depth_fname)
        depth_image = read_image(depth_path) / 255.

        if self.transform:
            if random.random() > 0.5:
                sem_imgs, depth_image = vflip(sem_imgs), vflip(depth_image)
            if random.random() > 0.5:
                sem_imgs, depth_image = hflip(sem_imgs), hflip(depth_image)

        return sem_imgs, depth_image