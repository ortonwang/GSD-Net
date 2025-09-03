""" Full assembly of the parts to form the complete network """

# import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
# from .unet_parts import *
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        # print(name)
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]
class DoubleConv(MetaModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            MetaConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            MetaBatchNorm2d(mid_channels, affine=bias),
            nn.ReLU(inplace=True),
            MetaConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            MetaBatchNorm2d(out_channels, affine=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(MetaModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, p=0.5, bias=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(p),
            DoubleConv(in_channels, out_channels, bias=bias)
        )
    def forward(self, x):
        # print('down')
        return self.maxpool_conv(x)


class Up(MetaModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, p=0.5, bias=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, bias=bias)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, bias=bias)
        self.p = p

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = nn.Dropout(self.p)(x)
        return self.conv(x)


class OutConv(MetaModule):
    def __init__(self, in_channels, out_channels, bias=True):
        super(OutConv, self).__init__()
        self.conv = MetaConv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.conv(x)

    def get_bias(self):
        return self.conv.bias

class UNet(MetaModule):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=True, p=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.inc = DoubleConv(n_channels, 16, bias=bias)
        self.down1 = Down(16, 32, p=0, bias=bias)
        self.down2 = Down(32, 64, p=0, bias=bias)
        self.down3 = Down(64, 128, p=0, bias=bias)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor, p=p, bias=bias)
        self.up1 = Up(256, 128 // factor, bilinear, p=0, bias=bias)
        self.up2 = Up(128, 64 // factor, bilinear, p=0, bias=bias)
        self.up3 = Up(64, 32 // factor, bilinear, p=0, bias=bias)
        self.up4 = Up(32, 16, bilinear, p=p, bias=bias)
        self.outc = OutConv(16, n_classes, bias=bias)

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits


class UNet_meta(MetaModule):
    def __init__(self,  bias=True, p=0):
        super(UNet_meta, self).__init__()
        self.conv1 = MetaConv2d(1, 100, kernel_size=3, padding=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = MetaConv2d(100, 1, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        return F.sigmoid(x)

class UNet_meta_smw(MetaModule):
    def __init__(self,  bias=True, p=0):
        super(UNet_meta_smw, self).__init__()
        self.conv1 = MetaConv2d(1, 100, kernel_size=3, padding=1, bias=bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = MetaConv2d(100, 1, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        return F.sigmoid(x)


if __name__ == '__main__':
    data = torch.randn(4,3,256,256).cuda()
    model = UNet(3,2).cuda()
    out = model(data)

    print('fsd')
