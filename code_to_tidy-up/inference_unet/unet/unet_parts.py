""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
# ajouter calcul de lignes/colonnes de pixel pourri a chaque étape du réseau

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential( # with padding = same -> output size = input, output trash pixels line/col = input trash pixels + (kernel size -1)
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding= 0, bias=False), 
            nn.BatchNorm2d(mid_channels),  
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding= 0, bias=False), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(  # with padding = same -> output size = input /2; output trash = roof(input trash /2) + (kernel size -1) 
            nn.MaxPool2d(2),   
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
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2): # with padding = same -> output size = input * 2, output trash pixels line/col = input trash pixels * 2 + (kernel size -1)
        #print(x1.size(),x2.size())
        x1 = self.up(x1) # output size = input * 2; output trash = input trash * 2
        #print(x1.size(),x2.size())
        diffY = x2.size()[2] - x1.size()[2]  #with padding = same -> normally zero because 64 - 64
        
        diffX = x2.size()[3] - x1.size()[3]  #with padding = same -> normally zero because 64 - 64
        #print(diffY,diffX)
        #diffX = x2.size()[3] - x1.size()[3]  # if change, need to adapt to the calculation of trash pixels
        marginY = diffY//2
        marginX = diffX//2
        #print(marginY,marginX)
        #print(x1.size(),x2[:,:,marginY:-1*marginY,marginX:-1*marginX].size())
        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2[:,:,marginY:-1*marginY,marginX:-1*marginX], x1], dim=1) 
        return self.conv(x)   #with padding = same ->  output size = input, output trash pixels line/col = input trash pixels + (kernel size -1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
