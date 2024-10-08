"""
train_data (torch.utils.data.Dataset),
test_data (torch.utils.data.Dataset),
and the model (torch.nn.Module) should be implemented here.

"""
import torch.nn as nn
import torch.utils.data
from typing import *
import flgo
from torchvision import datasets, transforms
import os
import warnings
import torch
from PIL import Image
from torchvision.datasets import VisionDataset
import torch.utils.data as tud
from torchvision.datasets.utils import download_and_extract_archive
import torch.nn.functional as F
root = os.path.join(flgo.benchmark.data_root, 'digits')

class MNISTM(VisionDataset):
    """MNIST-M Dataset.
    """

    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        # print(os.path.join(self.processed_folder, data_file))

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the MNIST-M data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class SyntheticDigits(VisionDataset):
    """Synthetic Digits Dataset.
    """

    resources = [
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz',
         'd0e99daf379597e57448a89fc37ae5cf'),
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz',
         '669d94c04d1c91552103e9aded0ee625')
    ]

    training_file = "synth_train.pt"
    test_file = "synth_test.pt"
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """Init Synthetic Digits dataset."""
        super(SyntheticDigits, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        print(os.path.join(self.processed_folder, data_file))

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        """Download the Synthetic Digits data."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

trans_usps = transforms.Compose([transforms.Resize([28,28]),transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans_svhn = transforms.Compose([transforms.Resize([28,28]),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans_mnist = transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans_syn = transforms.Compose([transforms.Resize([28,28]),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trans_mnistm = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trans_usps = trans_svhn = trans_mnist = trans_syn = trans_mnistm = transforms.Compose([transforms.Resize([28,28]),transforms.ToTensor()])

usps_train = datasets.USPS(root=root, train=True, download=True, transform=trans_usps)
usps_test = datasets.USPS(root=root, train=False, download=True, transform=trans_usps)
usps = tud.ConcatDataset([usps_train, usps_test])

svhn_train = datasets.SVHN(root=root, split='train', download=True, transform=trans_svhn)
svhn_test = datasets.SVHN(root=root, split='test', download=True, transform=trans_svhn)
svhn = tud.ConcatDataset([svhn_train, svhn_test])

mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=trans_mnist)
mnist_test = datasets.MNIST(root=root, train=False, download=True, transform=trans_mnist)
mnist = tud.ConcatDataset([mnist_train, mnist_test])

synthetic_train = SyntheticDigits(root=root, train=True, download=True, transform=trans_syn)
synthetic_test = SyntheticDigits(root=root, train=False, download=True, transform=trans_syn)
synthetic = tud.ConcatDataset([synthetic_train, synthetic_test])

mnistm_train = MNISTM(root=root, train=True, download=True, transform=trans_mnistm)
mnistm_test = MNISTM(root=root, train=False, download=True, transform=trans_mnistm)
mnistm = tud.ConcatDataset([mnistm_train, mnistm_test])

idx_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'subset_indices.pth')
datas = [usps, svhn, mnist, synthetic, mnistm]
if not os.path.exists(idx_file):
    data_sizes = [len(d) for d in datas]
    min_size = min(data_sizes)
    random_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    subset_indices = torch.stack([torch.randperm(n)[:min_size] for n in data_sizes])
    torch.save(subset_indices, idx_file)
    torch.random.set_rng_state(random_state)
else:
    subset_indices = torch.load(idx_file)
train_data = [torch.utils.data.Subset(datas[i], subset_indices[i]) for i in range(len(datas))]
val_data = None
test_data = None

def data_to_device(batch_data, device):
    return batch_data[0].to(device), batch_data[1].to(device)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

loss_fn = nn.CrossEntropyLoss()

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


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional layers with Batch Normalization and ReLU
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        x = self.bn4(self.conv4(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.bn5(self.conv5(x))
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten the output
        x = x.view(x.size(0), -1)  # Flatten
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation for final layer, softmax applied outside
        return x

def get_model(*args, **kwargs) -> torch.nn.Module:
    return AlexNet()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    domains = ['usps', 'svhn', 'mnist', 'synthetic', 'mnistm']
    t2i = transforms.Compose([transforms.ToPILImage(), ])
    WIDTH = 3
    COLS = 5
    for k, dk in enumerate(train_data):
        fig, ax = plt.subplots(nrows=WIDTH, ncols=WIDTH, sharex='all', sharey='all')
        ax = ax.flatten()
        for i in range(WIDTH**2):
            dk_train = dk
            idx = np.random.choice(range(len(dk_train)))
            img = t2i(dk_train[idx][0])
            ax[i].set_title(f"{int(dk_train[idx][1])}", loc='center',pad=0, fontsize=20, fontweight='bold')
            ax[i].imshow(img, interpolation='nearest')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.axis('off')
        # plt.tight_layout()
        # fig.suptitle(domains[k], fontsize=14, fontweight='bold')
        plt.savefig(f'tmp_{domains[k]}.png', dpi=300)
    imgs = [Image.open(f'tmp_{domain}.png') for k, domain in enumerate(domains)]
    n = len(imgs)
    ROWS = int(n/COLS)
    if n % COLS != 0: ROWS+=1
    fig, axs = plt.subplots(ROWS, COLS, figsize=(15,3),sharex='all', sharey='all')
    axs = axs.flatten()
    # 遍历图像并绘制
    for k, (ax, img) in enumerate(zip(axs, imgs)):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(f"Client-{domains[k]}", loc='center', pad=0, fontsize=11, fontweight='bold')
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