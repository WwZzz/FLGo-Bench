import torchvision.datasets.utils as tdu
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import torch.utils.data as tud
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
root = os.path.dirname(__file__)
root = '/data/wz/camelyon17'

class Camelyon17(Dataset):

    url = "https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/"
    zipfile = "camelyon17_v1.0.tar.gz" # filesize = 10GB

    def __init__(self, root, site, split='train', transform=None):
        assert split in ['train', 'test']
        assert int(site) in [0,1,2,3,4] # five hospital
        self.root = root
        meta_path = os.path.join(root, 'metadata.csv')
        zip_path = os.path.join(root, self.zipfile)
        if not os.path.exists(meta_path):
            if not os.path.exists(zip_path):
                tdu.download_url('https://worksheets.codalab.org/rest/bundles/0xe45e15f39fb54e9d9e919556af67aabe/contents/blob/', root, 'camelyon17_v1.0.tar.gz')
            tdu.extract_archive(os.path.join(root, 'camelyon17_v1.0.tar.gz'))
        meta_data = pd.read_csv(meta_path)
        meta_data = meta_data[meta_data['node'] == site]
        meta_data = meta_data[meta_data['split'] == (0 if split == 'train' else 1)].to_numpy()
        self.imgs = [os.path.join(root, 'patches', "_".join(["patient", "{:03d}".format(row[1]), 'node', f'{site}']), "_".join(["patch","patient", "{:03d}".format(row[1]), 'node', f'{site}',"x",f"{row[3]}", "y",f"{row[4]}.png"  ])) for row in meta_data]
        self.labels = torch.from_numpy(meta_data[:, 5].astype(np.int64))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

train_data = [tud.ConcatDataset([Camelyon17(root, i, split='train', transform=transforms.ToTensor()), Camelyon17(root, i, split='test', transform=transforms.ToTensor())]) for i in range(5)]
test_data = None
val_data = None


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, **kwargs):
        super(_DenseLayer, self).__init__()
        self.add_module('bn1', nn.BatchNorm2d(num_input_features, affine=False, track_running_stats=False)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('bn2', nn.BatchNorm2d(bn_size * growth_rate, affine=False, track_running_stats=False)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, **kwargs):
        super(_DenseBlock, self).__init__()

        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, **kwargs):
        super(_Transition, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(num_input_features, affine=False, track_running_stats=False))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, input_shape=(3, 96, 96), growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=2, **kwargs):

        super(DenseNet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn0', nn.BatchNorm2d(num_init_features, affine=False, track_running_stats=False)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i == 0:
                self.features.add_module('zero_padding', nn.ZeroPad2d(2))
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        self.features.add_module('bn5', nn.BatchNorm2d(num_features, affine=False, track_running_stats=False))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

loss_fn = nn.CrossEntropyLoss()

def data_to_device(batch_data, device):
    return batch_data[0].to(device), batch_data[1].to(device)

def eval(model, data_loader, device) -> dict:
    model.eval()
    model.to(device)
    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    for batch_id, batch_data in enumerate(data_loader):
        batch_data = data_to_device(batch_data, device)
        outputs = model(batch_data[0])
        batch_mean_loss = loss_fn(outputs, batch_data[-1]).item()
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()
        num_correct += correct.item()
        num_samples += len(batch_data[-1])
        total_loss += batch_mean_loss * len(batch_data[-1])
    return {'accuracy': 1.0 * num_correct / num_samples, 'loss': total_loss / num_samples}

def compute_loss(model, batch_data, device) -> dict:
    tdata = data_to_device(batch_data, device)
    outputs = model(tdata[0])
    loss = loss_fn(outputs, tdata[-1])
    return {'loss': loss}


def get_model():
    return DenseNet() # Dense121

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    domains = ['A', 'B', 'C', 'D', 'E']
    classes = ['Normal', 'Tumor']
    t2i = transforms.Compose([transforms.ToPILImage(), ])
    WIDTH = 3
    COLS = 5
    fig_size = (17,3)
    subtitle_size = 15
    suptitle_size = 20
    for k, dk in enumerate(train_data):
        fig, ax = plt.subplots(nrows=WIDTH, ncols=WIDTH, sharex='all', sharey='all')
        ax = ax.flatten()
        for i in range(WIDTH**2):
            dk_train = dk.datasets[0]
            idx = np.random.choice(range(len(dk_train)))
            img = t2i(dk_train[idx][0])
            ax[i].set_title(classes[int(dk_train[idx][1])], loc='center',pad=0, fontsize=subtitle_size, fontweight='bold')
            ax[i].imshow(img, interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.axis('off')
        # plt.tight_layout()
        # fig.suptitle(domains[k], fontsize=14, fontweight='bold')
        plt.savefig(f'tmp_{domains[k]}.png', dpi=600)
    imgs = [Image.open(f'tmp_{domain}.png') for k, domain in enumerate(domains)]
    n = len(imgs)
    ROWS = int(n/COLS)
    if n % COLS != 0: ROWS+=1
    fig, axs = plt.subplots(ROWS, COLS, figsize=fig_size,sharex='all', sharey='all')
    axs = axs.flatten()
    # 遍历图像并绘制
    for k, (ax, img) in enumerate(zip(axs, imgs)):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(f"Hospital-{domains[k]}", loc='center', pad=0, fontsize=suptitle_size, fontweight='bold')
        ax.text(0.5, -0.005, f'size={len(train_data[k])}', ha='center', va='top', transform=ax.transAxes)
        ax.axis('off')  # 不显示坐标轴
    # 显示拼接后的图形
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0., hspace=0.)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('res.png', dpi=300)
    [os.remove(f'tmp_{domain}.png') for k, domain in enumerate(domains)]
    # plt.show()
    # img = data[0].datasets[0][0][0]
    # img = tensor2img(img)
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.show()
    print('ok')