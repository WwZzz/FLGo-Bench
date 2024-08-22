# default_model.py
import torch

from ..config import get_model
from flgo.utils.fmodule import FModule
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import random
import numpy as np
import cv2
import albumentations as A

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)



class AugmentDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = A.Compose([A.RandomRotate90(), A.HorizontalFlip(p=0.75), A.VerticalFlip(p=0.75)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        res = self.transform(image=np.transpose(img.numpy(), (2,1,0)), mask=np.transpose(label.numpy(), (2,1,0)))
        return torch.from_numpy(np.transpose(res['image'], (2,1,0))), torch.from_numpy(np.transpose(res['mask'], (2,1,0)))

def init_dataset(object):
    if 'Client' in object.get_classname():
        object.train_data = AugmentDataset(object.train_data)
