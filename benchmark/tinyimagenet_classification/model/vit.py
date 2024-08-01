import torch
import torch.nn as nn
import numpy as np
import flgo.utils.fmodule as fmodule

class Model(fmodule.FModule):
    def __init__(self):
        super().__init__()
        self.model = VisionTransformer()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

def init_local_module(object):
    pass

def init_global_module(object):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)

class ResidualLayer(nn.Module):
    def __init__(self, ch, kernel_size, activation=nn.ELU()):
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=2)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.activation(x1)
        res = self.conv2(x1)
        return x + res


def to_pair(x):
    if type(x) == tuple:
        return x
    return (x, x)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """ Multihead self-attention layer. """

    def __init__(self, dim, heads, dimheads=None):
        super().__init__()
        self.heads = heads
        self.dimheads = dim // heads if not dimheads else dimheads
        self.dim_inner = self.dimheads * heads * 3  # for qkv
        self.scale = self.dimheads ** -0.5

        # layers
        self.norm = nn.LayerNorm(dim)
        self.embed_qkv = nn.Linear(dim, self.dim_inner, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Linear(heads * self.dimheads, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        # querys, keys, values
        qkv = self.embed_qkv(x)
        qkv = qkv.view(x.shape[0], self.heads, x.shape[1], 3, self.dimheads)  # (b, h, n, qkv, dimh)
        q, k, v = [x.squeeze() for x in torch.split(qkv, 1, dim=-2)]

        # compute attention per head
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)

        # stack output of heads
        out = out.transpose(-3, -2).flatten(start_dim=-2)
        out = self.to_out(out)
        return out


class TransformerEncoder(nn.Module):
    """ Single transformer encoder unit. """

    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.attention = Attention(dim, heads)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer from the paper 'An Image is Worth 16x16 Words: Transformers for Image Recognition
    at Scale', Dosovitskiy et al., 2021. The implemented model is a hybrid version if convs are neabled.
    With num_convs=0 and droprate=0 it becomes a regular ViT.
    """

    def __init__(self, img_size=64, patch_size=8, num_classes=200, dim=256, depth=12, heads=12, mlp_dim=256,
                 channels=3, num_convs=0, droprate=0.):
        super().__init__()
        self.img_height, self.img_width = to_pair(img_size)
        self.patch_height, self.patch_width = to_pair(patch_size)
        features_height, features_width = self.img_height // (2 ** num_convs), self.img_width // (2 ** num_convs)
        self.num_patches = features_height // self.patch_height * features_width // self.patch_width
        assert self.img_width % self.patch_width == 0 and self.img_height % self.patch_height == 0

        # define layers
        conv_layers = [nn.Identity()]
        for i in range(num_convs):
            conv_layers.extend([ResidualLayer(ch=channels, kernel_size=5),
                                nn.Conv2d(channels, channels * 2, kernel_size=5, padding=2),
                                nn.ELU(),
                                nn.MaxPool2d(2, 2)])
            channels = channels * 2
        self.convs = nn.Sequential(*conv_layers)

        patch_dims = self.patch_width * self.patch_height * channels
        self.patch_embedding = nn.Linear(patch_dims, dim)
        self.pos_embedding = nn.Parameter(torch.zeros(self.num_patches, dim, requires_grad=True))

        encoder_modules = []
        for i in range(depth):
            encoder_modules.append(TransformerEncoder(dim, heads, mlp_dim))
        self.transformer_encoders = nn.Sequential(*encoder_modules)

        self.dropout = nn.Dropout(droprate)
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes))

    def _to_patches(self, x):
        x = self.convs(x)
        bs, ch, _, _ = x.shape
        ph, pw, np = self.patch_height, self.patch_width, self.num_patches
        patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
        patches = patches.contiguous().view(bs, ch, np, ph, pw)
        patches = patches.permute(0, 2, 3, 4, 1)
        return patches

    def forward(self, x):
        # patch encoding
        patches = self._to_patches(x)
        patches_flat = torch.flatten(patches, start_dim=2)

        # pos embedding could also be concatenated
        z = self.patch_embedding(patches_flat) + self.pos_embedding
        class_tokens = torch.zeros([z.shape[0], 1, z.shape[2]]).to(z.get_device())
        z = torch.cat((class_tokens, z), dim=1)

        # transformers and prediction
        z = self.transformer_encoders(z)
        ct_drop = self.dropout(z[:, 0])
        return self.prediction_head(ct_drop)