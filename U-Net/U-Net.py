import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class UNet(pl.LightningModule):
    def __init__(self, n_in_features, n_out_features, hidden=4, bilinear=False):
        super(UNet, self).__init__()
        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        self.bilinear = bilinear
        self.hidden = hidden
        self.inc = DoubleConv(n_in_features, hidden)
        self.down1 = Down(hidden, hidden*2)
        self.down2 = Down(hidden*2, hidden*4)
        self.down3 = Down(hidden*4, hidden*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(hidden*8, hidden*16 // factor)
        self.up1 = Up(hidden*16, hidden*8 // factor, bilinear)
        self.up2 = Up(hidden*8, hidden*4 // factor, bilinear)
        self.up3 = Up(hidden*4, hidden // factor, bilinear)
        #self.up4 = Up(hidden*2, hidden, bilinear)
        self.outc = OutConv(hidden, n_out_features)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        #x = self.up4(x, x1)
        output = self.outc(x)
        return output

    """ Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        # ()
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2],
                   "constant", 0)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
