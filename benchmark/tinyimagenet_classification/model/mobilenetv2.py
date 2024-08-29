from flgo.utils import fmodule
import torchvision.models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as tud
class Model(fmodule.FModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2()
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 200)

    def forward(self, x):
        return self.model(x)

class AugmentDataset(tud.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose(
            [transforms.Resize((224,224)), transforms.RandomRotation(20), transforms.RandomHorizontalFlip(0.5)])

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