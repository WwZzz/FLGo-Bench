from ..config import get_model
from flgo.utils.fmodule import FModule
import torchvision.transforms as transforms
import torch.utils.data as tud
class Model(FModule):
    def __init__(self):
        super().__init__()
        self.model = get_model()
        if hasattr(self.model, 'compute_loss'):
            self.compute_loss = self.model.compute_loss

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)

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