import pytorch_lightning as pl

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
