"""
This model is copied from the github repo of FedDyn
(https://github.com/alpemreacar/FedDyn/blob/master/utils_models.py)
"""

from flgo.utils import fmodule
import torchvision.models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as tud
class Model(fmodule.FModule):
    def __init__(self):
        super().__init__()
        resnet18 = torchvision.models.resnet18(norm_layer=lambda x: nn.GroupNorm(2, x))
        resnet18.avgpool = nn.AdaptiveAvgPool2d(1)
        num_features = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_features, 200)
        self.model = resnet18

    def forward(self, x):
        return self.model(x)

class AugmentDataset(tud.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose(
            [transforms.RandomRotation(20), transforms.RandomHorizontalFlip(0.5)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, label = self.dataset[item]
        return self.transform(img), label

def init_dataset(object):
    if 'Client' in object.get_classname():
        object.train_data = AugmentDataset(object.train_data)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)