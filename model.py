import torch
import torch.nn as nn
import math
from math import exp
from torch.autograd import Variable
import torch.nn.functional as F

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def avg_filter(img1, window_size, sigma):
    channel=1
    window = create_window(window_size, channel, sigma)
    window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    return mu1

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()

        # Contracting path
        self.FE1 = DoubleConv(1, 16)
        self.down_conv_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FE2 = DoubleConv(16, 32)
        self.down_conv_2 = nn.MaxPool2d(kernel_size=2, stride=2)        
        self.FE3 = DoubleConv(32, 32)

    def forward(self, x):

        # Contracting path
        x1 = self.FE1(x)
        x = self.down_conv_1(x1)

        x2 = self.FE2(x)
        x = self.down_conv_2(x2)

        # Bottom
        x = self.FE3(x)

        return x, x2, x1
    
class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()

        # Contracting path
        self.FE1 = DoubleConv(1, 16)
        self.down_conv_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FE2 = DoubleConv(16, 32)
        self.down_conv_2 = nn.MaxPool2d(kernel_size=2, stride=2)        
        self.FE3 = DoubleConv(32, 32)

    def forward(self, x):

        # Contracting path
        x1 = self.FE1(x)
        x = self.down_conv_1(x1)

        x2 = self.FE2(x)
        x = self.down_conv_2(x2)

        # Bottom
        x = self.FE3(x)

        return x, x2, x1

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Contracting path
        self.FE1 = DoubleConv(2, 16)
        self.down_conv_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.FE2 = DoubleConv(16, 32)
        self.down_conv_2 = nn.MaxPool2d(kernel_size=2, stride=2)        
        self.FE3 = DoubleConv(96, 32)

        # Expansive path
        self.up_conv_1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.FE4 = DoubleConv(128, 32)
        self.up_conv_2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.FE5 = DoubleConv(64, 1)
        self.aux1 = Decoder1()
        self.aux2 = Decoder2()



    def forward(self, imag1, imag2):
        
        
        imag1_base = avg_filter(imag1,11, 1)
        imag2_base = avg_filter(imag2,11, 1) 

        fused_base,_ = torch.max(torch.cat((imag1_base,imag2_base),dim=1),dim=1)
        fused_base = fused_base.unsqueeze(1)

        fused_base1, fused_base2, fused_base3 = self.aux1(fused_base)


        imag1_detail = imag1 - imag1_base
        imag2_detail = imag2 - imag2_base
        fused_detail = imag1_detail*1/2 + imag2_detail*1/2
        fused_detail1, fused_detail2, fused_detail3 = self.aux1(fused_detail)

        
        # Contracting path
        x1 = self.FE1(torch.cat([imag1, imag2], dim=1))
        x = self.down_conv_1(x1)

        x2 = self.FE2(x)
        x = self.down_conv_2(x2)


        # Bottom
        x = self.FE3(torch.cat([x, fused_base1, fused_detail1], dim = 1))

        # Expansive path
        x = self.up_conv_1(x)
        x = torch.cat([x2, x, fused_base2, fused_detail2], dim=1)
        
        

        x = self.FE4(x)

        x = self.up_conv_2(x)
        x = torch.cat([x1, x, fused_base3, fused_detail3], dim=1)


        x = self.FE5(x)

        return x
    
class Resblock(nn.Module):
    def __init__(self, in_channels,mid_channels,  out_channels):
        super(Resblock, self).__init__()
        
        self.con1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.con = nn.ReLU(inplace=True)
        self.con2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        self.con3 =  nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.con(self.con3(self.con(self.con2(self.con(self.con1(x))))))+x
    
# 定义SKBlock
class SKBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, stride=1):
        super(SKBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, f1, f2):
   
        u = self.avg_pool(f1+f2)
        u = u.view(u.size(0), -1)
        z = self.fc(u)
        z = z.view(z.size(0), -1, 1, 1)
        z = self.softmax(z)
        s1 = (z * f1)
        s2 = ((1 - z) * f2)

        return s1, s2, (z * f1) + ((1 - z) * f2)


# 定义图像融合网络
class ImageFusionNetwork(nn.Module):
    def __init__(self):
        super(ImageFusionNetwork, self).__init__()
        self.resnet1 = Resblock(16, 16, 16)
        self.resnet2 = Resblock(16, 16, 16)
        self.resnet3 = Resblock(16, 16, 16)
        self.resnet4 = Resblock(16, 16, 16)
        self.resnet5 = Resblock(16, 16, 16)
        self.resnet6 = Resblock(16, 16, 16)
        self.sk_block = SKBlock(16)  # 假设ResNet-50输出通道数为2048
        self.conv1 = nn.Conv2d(1, 16, kernel_size=1)  # 降维   
        self.conv2 = nn.Conv2d(1, 16, kernel_size=1)  # 降维   

        self.convf = nn.Conv2d(16, 1, kernel_size=1)  # 降维        
     

    def forward(self, input1, input2):
        # 对两个输入图像分别提取特征        
        x1 = self.resnet1(self.conv1(input1))
        x2 = self.resnet1(self.conv2(input2))

        # 将两个特征图进行特征融合        

        s1, s2, x3 = self.sk_block(x1, x2)

        # 通过卷积层获得融合后的输出        
        x = self.convf(self.resnet3(x3))


        return x
    


# 创建模型
"""
model = ImageFusionNetwork().cuda()
a = torch.randn(1, 1, 702, 402).cuda()
b = torch.randn(1, 1, 702, 402).cuda()
b = model(a, b)
print(b.shape)
"""