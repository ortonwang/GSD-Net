""" Full assembly of the parts to form the complete network """

# import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .unet_parts import *


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bias=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(mid_channels, affine=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels, affine=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, p=0.5, bias=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(p),
            DoubleConv(in_channels, out_channels, bias=bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
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



class UNet_urpc(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=True, p=0):
        super(UNet_urpc, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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

        self.outc1 = nn.Conv2d(16, n_classes, kernel_size=3, padding=1)
        self.outc2 = nn.Conv2d(32, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xd3 = self.up1(x5, x4)
        xd2 = self.up2(xd3, x3)
        xd1 = self.up3(xd2, x2)
        xd = self.up4(xd1, x1)
        logits = self.outc(xd)

        xdd1 = self.outc1(xd1)
        xdd1 = torch.nn.functional.interpolate(xdd1, (256,256))

        xdd2 = self.outc2(xd2)
        xdd2 = torch.nn.functional.interpolate(xdd2, (256, 256))

        return logits ,xdd1 ,xdd2

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.conv(x)

    def get_bias(self):
        return self.conv.bias

class UNet_drop(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=True, p=0):
        super(UNet_drop, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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
        self.Drop1 = nn.Dropout(p=0.05)
        self.Drop2 = nn.Dropout(p=0.07)
        self.Drop3 = nn.Dropout(p=0.09)
        self.Drop4 = nn.Dropout(p=0.12)
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.Drop1(x1)
        x2 = self.down1(x1)
        x2 = self.Drop2(x2)
        x3 = self.down2(x2)
        x3 = self.Drop3(x3)
        x4 = self.down3(x3)
        x4 = self.Drop4(x4)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=True, p=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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



class UNet_PNP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=True, p=0):
        super(UNet_PNP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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
        self.PNP_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=bias),
            nn.BatchNorm2d(16, affine=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1, bias=bias),
            torch.nn.Sigmoid()
        )

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
        prob = self.PNP_conv(x)
        return {'logits': logits, 'prob': prob}

class UNet_PNP2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=True, p=0):
        super(UNet_PNP2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.inc = DoubleConv(n_channels, 16, bias=bias)
        self.inc2 = DoubleConv(n_channels+1, 16, bias=bias)
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
        self.outc2 = nn.Sequential(OutConv(16, n_classes, bias=bias),torch.nn.Sigmoid())

    def forward(self, x_in,gt):
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        x_gt = torch.cat([x_in,gt],1)
        x1 = self.inc2(x_gt)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        prob = self.outc2(x)
        return {'logits': logits, 'prob': prob}

class UNet_PNP27(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, bias=True, p=0):
        super(UNet_PNP27, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.inc = DoubleConv(n_channels, 16, bias=bias)
        self.inc2 = DoubleConv(n_channels+1, 16, bias=bias)
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
        self.outc2 = nn.Sequential(OutConv(16, n_classes, bias=bias),torch.nn.Sigmoid())
        self.PNP_conv = nn.Sequential(
            nn.Conv2d(n_classes+1, 16, kernel_size=1, bias=bias),
            nn.BatchNorm2d(16, affine=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=bias),
            torch.nn.Sigmoid()
        )
    def forward(self, x_in,gt):
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits_soft = torch.softmax(logits,1)
        logit_gt = torch.cat([logits_soft, gt], 1)
        prob = self.PNP_conv(logit_gt)
        return {'logits': logits, 'prob': prob}






if __name__ == '__main__':
    data = torch.randn(4,3,256,256)
    model = UNet_PNP(3,2)
    out = model(data)

    print('fsd')
