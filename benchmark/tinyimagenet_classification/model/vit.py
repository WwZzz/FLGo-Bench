import torch.nn as nn
import flgo.utils.fmodule as fmodule
import torchvision.models as models

class Model(fmodule.FModule):
    def __init__(self):
        super().__init__()
        model = models.vit_b_16()
        # model = models.vit_b_16(models.ViT_B_16_Weights)
        model.head = nn.Linear(model.heads.head.in_features, 200)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)
