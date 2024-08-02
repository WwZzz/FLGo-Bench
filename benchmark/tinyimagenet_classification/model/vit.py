import torch.nn as nn
import flgo.utils.fmodule as fmodule
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as tud
class Model(fmodule.FModule):
    def __init__(self):
        super().__init__()
        model = models.vit_b_16()
        # model = models.vit_b_16(models.ViT_B_16_Weights)
        model.head = nn.Linear(model.heads.head.in_features, 200)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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
