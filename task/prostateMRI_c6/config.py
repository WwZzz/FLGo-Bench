import sys, os


import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import SimpleITK as sitk
import random
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.datasets.utils as tdu
import sys, os
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
root = os.path.dirname(os.path.abspath(__file__))
def convert_from_nii_to_png(img):
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg

class Prostate(Dataset):
    url = "https://github.com/med-air/HarmoFL/raw/main/data/prostate/{}"
    def __init__(self, root, site, split='train', transform=None):
        channels = {'BMC': 3, 'BIDMC': 3, 'RUNMC': 3, 'I2CVB': 3, 'HK': 3, 'UCL': 3}
        assert site in list(channels.keys())
        self.split = split
        self.root = root
        if not os.path.exists(os.path.join(root, f'HK')):
            if not os.path.exists(os.path.join(root, 'Processed_data_nii.zip')):
                tdu.download_file_from_google_drive('1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-', root)
            tdu.extract_archive(os.path.join(root, 'Processed_data_nii.zip'), root, remove_finished=True)
        images, labels = [], []
        sitedir = os.path.join(root, site)
        ossitedir = os.listdir(sitedir)
        case_segs = [s for s in ossitedir if 'seg' in s.lower()]
        for case_seg in case_segs:
            case = case_seg.split('_')[0]
            samplerdir = os.path.join(sitedir, case_seg)
            imgdir = os.path.join(sitedir, f'{case}.nii.gz')
            label_v = sitk.ReadImage(samplerdir)
            image_v = sitk.ReadImage(imgdir)
            label_v = sitk.GetArrayFromImage(label_v)
            label_v[label_v > 1] = 1
            image_v = sitk.GetArrayFromImage(image_v)
            image_v = convert_from_nii_to_png(image_v)
            for i in range(1, label_v.shape[0] - 1):
                label = np.array(label_v[i, :, :])
                if (np.all(label == 0)):
                    continue
                image = np.array(image_v[i - 1:i + 2, :, :])
                image = np.transpose(image, (1, 2, 0))
                labels.append(label)
                images.append(image)
        labels = np.array(labels).astype(int)
        images = np.array(images)
        self.images = images
        self.labels = labels
        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

sites = ['BMC','UCL','BIDMC', 'RUNMC', 'HK', 'I2CVB']
train_data = [Prostate(root, s, transform=transforms.ToTensor()) for s in sites]
test_data = None
val_data = None

def _block(in_channels, features, name, affine=True, track_running_stats=True):
    bn_func = nn.BatchNorm2d

    return nn.Sequential(
        OrderedDict(
            [
                (
                    name + "_conv1",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn1", bn_func(num_features=features, affine=affine,
                                        track_running_stats=track_running_stats)),
                (name + "_relu1", nn.ReLU(inplace=True)),
                (
                    name + "_conv2",
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=features,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                ),
                (name + "_bn2", bn_func(num_features=features, affine=affine,
                                        track_running_stats=track_running_stats)),
                (name + "_relu2", nn.ReLU(inplace=True)),
            ]
        )
    )

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, aug_method=None):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)

        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)

        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)

        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)

        bottleneck = self.bottleneck(enc4_)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)

        return dec1

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):
            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred == i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt == i] = 1

            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)

            union = each_pred.view(batch_size, -1).sum(1) + each_gt.view(batch_size, -1).sum(1)
            dice = (2. * intersection) / (union + 1e-5)

            all_dice += torch.mean(dice)

        return all_dice * 1.0 / num_class

    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred, dim=1)

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]

        bg = torch.zeros_like(gt)
        bg[gt == 0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt == 1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)

        loss = 0
        smooth = 1e-5

        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...])
            y_sum = torch.sum(label[:, i, ...])
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, pred, gt):
        ce = self.ce(pred, gt.squeeze(axis=1).long())
        return (ce + self.dice(pred, gt)) / 2

loss_fn = JointLoss()

def data_to_device(batch_data, device):
    return batch_data[0].to(device), batch_data[1].to(device)

def compute_loss(model, batch_data, device) -> dict:
    tdata = data_to_device(batch_data, device)
    outputs = model(tdata[0])
    loss = loss_fn(outputs, tdata[-1])
    return {'loss': loss}

def dice_func(output, label):
    softmax_pred = torch.nn.functional.softmax(output, dim=1)
    seg_pred = torch.argmax(softmax_pred, dim=1)
    all_dice = 0
    label = label.squeeze(dim=1)
    batch_size = label.shape[0]
    num_class = softmax_pred.shape[1]
    for i in range(num_class):
        each_pred = torch.zeros_like(seg_pred)
        each_pred[seg_pred == i] = 1
        each_gt = torch.zeros_like(label)
        each_gt[label == i] = 1
        intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
        union = each_pred.view(batch_size, -1).sum(1) + each_gt.view(batch_size, -1).sum(1)
        dice = (2. * intersection) / (union + 1e-5)
        all_dice += torch.mean(dice)
    return all_dice.item() * 1.0 / num_class

def eval(model, data_loader, device) -> dict:
    model.eval()
    model.to(device)
    total_loss = 0.0
    num_samples = 0
    for batch_id, batch_data in enumerate(data_loader):
        batch_data = data_to_device(batch_data, device)
        outputs = model(batch_data[0])
        batch_loss = dice_func(outputs, batch_data[1])
        num_samples += len(batch_data[-1])
        total_loss += batch_loss * len(batch_data[-1])
    return {'Dice': total_loss / num_samples}

def get_model():
    return UNet()

if __name__ == '__main__':
    # model = UNet()
    # x = train_data[0][0][0].unsqueeze(0)
    # y = model(x)
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    t2i = transforms.Compose([transforms.ToPILImage(), ])
    WIDTH = 2
    COLS = 3
    domains = sites
    for k, dk in enumerate(train_data):
        fig, ax = plt.subplots(nrows=WIDTH, ncols=WIDTH, sharex='all', sharey='all')
        ax = ax.flatten()
        for i in range(WIDTH**2):
            dk_train = dk
            idx = np.random.choice(range(len(dk_train)))
            img = t2i(dk_train[idx][0])
            # ax[i].set_title(classes[int(dk_train[idx][1])], loc='center',pad=0, fontsize=11, fontweight='bold')
            ax[i].imshow(img)
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
    fig, axs = plt.subplots(ROWS, COLS, figsize=(5,3),sharex='all', sharey='all')
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
