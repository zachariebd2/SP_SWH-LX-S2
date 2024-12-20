""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from math import ceil


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        
        

    def forward(self, x):
        #print("x",x.size()) 
        x1 = self.inc(x) # with padding = same -> output size = input, output trash pixels line/col = input trash pixels + (kernel size -1)
        #print("x1 2conv",x1.size())
        x2 = self.down1(x1) #with padding = same -> output size = input /2; output trash = ceil(input trash /2) + (kernel size -1) 
        #print("x2 down1",x2.size())
        x3 = self.down2(x2)
        #print("x3 down2",x3.size())
        x4 = self.down3(x3)
        #print("x4 down3",x4.size())
        x5 = self.down4(x4)
        #print("x5 down4",x5.size())
        x = self.up1(x5, x4) #with padding = same -> output size = input * 2, output trash  = input trash * 2 + (kernel size -1)
        #print("x6 up1",x.size())
        x = self.up2(x, x3)
        #print("x7 up2",x.size())
        x = self.up3(x, x2)
        #print("x8 up3",x.size())
        x = self.up4(x, x1)
        #print("x9 up4",x.size())
        logits = self.outc(x) #with padding = same -> output size = input, output trash pixels line/col = input trash pixels 
        #print("logits outconv",logits.size())
        
        return logits 
