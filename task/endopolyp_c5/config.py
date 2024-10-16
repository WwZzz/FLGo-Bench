import flgo
import torchvision.datasets.utils as tdu
import os
import torchvision.transforms as transforms
import os
import torch.utils.data as tud
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

root = os.path.join(flgo.benchmark.data_root, 'endopolyp')

class PolypDataset(tud.Dataset):
    sites = ['Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-ClinicDB', 'CVC-300']
    def __init__(self, root, site, transform=None, target_transform=None):
        assert site in self.sites
        self.root = root
        self.site = site
        self.transform = transform
        self.target_transform = target_transform
        self.dirs = [os.path.join(root, 'TrainDataset'), os.path.join(root, 'TestDataset')]
        if not os.path.exists(os.path.join(self.root, 'TrainDataset')):
            if not os.path.exists(os.path.join(self.root, 'TrainDataset.zip')):
                tdu.download_file_from_google_drive('1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb', root, )
            tdu.extract_archive(os.path.join(self.root, 'TrainDataset.zip'), remove_finished=True)
        if not os.path.exists(os.path.join(self.root, 'TestDataset')):
            if not os.path.exists(os.path.join(self.root, 'TestDataset.zip')):
                tdu.download_file_from_google_drive('1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao', root, )
            tdu.extract_archive(os.path.join(self.root, 'TestDataset.zip'), remove_finished=True)
        # construct img list
        self.img_dir = os.path.join(self.root, 'TestDataset', site, 'images')
        self.img_list = [os.path.join(self.img_dir, r) for r in os.listdir(self.img_dir)]
        self.mask_list = [s.replace('images', 'masks') for s in self.img_list]
        if self.site in ['Kvasir', 'CVC-ClinicDB'] :
            train_img_dir = os.path.join(self.root, 'TrainDataset', 'image')
            f = (lambda x:len(x)>10) if self.site=='Kvasir' else (lambda x:len(x)<=10)
            train_img_list = [os.path.join(train_img_dir, r) for r in os.listdir(train_img_dir) if f(r)]
            train_mask_list = [s.replace('image', 'masks') for s in train_img_list]
            self.img_list.extend(train_img_list)
            self.mask_list.extend(train_mask_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB').resize((384, 384), Image.LANCZOS)
        target = Image.open(self.mask_list[index])
        if target.mode == 'RGB':
            target = target.convert('L')
        target = target.resize((384, 384))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

train_data = [PolypDataset(root, s, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), target_transform=transforms.ToTensor()) for s in PolypDataset.sites]
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
        batch_dice = dice_func(outputs, batch_data[1])
        num_samples += len(batch_data[-1])
        total_loss += batch_dice * len(batch_data[-1])
    return {'Dice': total_loss / num_samples}

def get_model():
    return UNet()

if __name__ == '__main__':
    data0 = train_data[0]
    x,y = data0[0]
    model = UNet()
    yp = model(x.unsqueeze(0))
    loss = loss_fn(yp, y.unsqueeze(0))
    print(yp.shape)
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    class UnNormalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            """
            Args:
                tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            Returns:
                Tensor: Normalized image.
            """
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
                # The normalize code -> t.sub_(m).div_(s)
            return tensor
    t2i = transforms.Compose([UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), transforms.ToPILImage(), ])
    t2i2 =  transforms.ToPILImage()
    WIDTH = 3
    COLS = 5
    domains = PolypDataset.sites
    for k, dk in enumerate(train_data):
        fig, ax = plt.subplots(nrows=WIDTH, ncols=WIDTH, sharex='all', sharey='all')
        ax = ax.flatten()
        for i in range(WIDTH ** 2):
            dk_train = dk
            idx = np.random.choice(range(len(dk_train)))
            img = t2i(dk_train[idx][0])
            label = t2i2(dk_train[idx][1])
            colormap = plt.get_cmap('jet', np.max(label) + 1)
            label_color = colormap(label)
            # ax[i].set_title(classes[int(dk_train[idx][1])], loc='center',pad=0, fontsize=11, fontweight='bold')
            ax[i].imshow(img)
            ax[i].imshow(label_color, alpha=0.5)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.axis('off')
        # plt.tight_layout()
        # fig.suptitle(domains[k], fontsize=14, fontweight='bold')
        plt.savefig(f'tmp_{domains[k]}.png', dpi=300)
    imgs = [Image.open(f'tmp_{domain}.png') for k, domain in enumerate(domains)]
    n = len(imgs)
    ROWS = int(n / COLS)
    if n % COLS != 0: ROWS += 1
    fig, axs = plt.subplots(ROWS, COLS, figsize=(10, 2), sharex='all', sharey='all')
    axs = axs.flatten()
    # 遍历图像并绘制
    for k, (ax, img) in enumerate(zip(axs, imgs)):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(f"Client-{domains[k]}", loc='center', pad=0, fontsize=8, fontweight='bold')
        ax.text(0.5, -0.03, f'size={len(train_data[k])}', ha='center', va='top', transform=ax.transAxes)
        ax.axis('off')  # 不显示坐标轴
    # 显示拼接后的图形
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0., hspace=0.)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig('res.png', dpi=300)
    [os.remove(f'tmp_{domain}.png') for k, domain in enumerate(domains)]
    print('Done')