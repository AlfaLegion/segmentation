import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchsummary import summary
import Loss.Losses as Losses
import DataLoader
from torch.autograd import Variable
import torchvision.utils as v_utils
from PIL import Image



class ConvBlock(nn.Module):
    def __init__(self,input_channels,out_channels,padding):
        super(ConvBlock,self).__init__()

        self.conv_1=nn.Conv2d(input_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn_1=nn.BatchNorm2d(out_channels)

        self.relu=nn.ReLU(inplace=True)

        self.conv_2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn_2=nn.BatchNorm2d(out_channels)

    def forward(self, input):

        input=self.conv_1(input)
        input=self.bn_1(input)
        input=self.relu(input)

        input=self.conv_2(input)
        input=self.bn_2(input)
        input=self.relu(input)

        return input

class Bottleneck(nn.Module):
    def __init__(self,input_channels,out_channels,padding):
        super(Bottleneck,self).__init__()

        self.conv_1=nn.Conv2d(input_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn_1=nn.BatchNorm2d(out_channels)
        self.relu_1=nn.ReLU(inplace=True)

        self.conv_2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn_2=nn.BatchNorm2d(out_channels)

    def forward(self,input):
        input=self.conv_1(input)
        input=self.bn_1(input)
        input=self.relu_1(input)
        input=self.conv_2(input)
        input=self.bn_2(input)
        
        return input

class DeconvBnRelu(nn.Module):
    def __init__(self,input_channels,out_channels,padding,out_padding):
        super(DeconvBnRelu,self).__init__()

        self.deconv1=nn.ConvTranspose2d(input_channels,out_channels,kernel_size=3, stride=2, padding=1,output_padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu1=nn.ReLU(inplace=True)
    
    def forward(self,input):

        input=self.deconv1(input)
        input=self.bn1(input)
        input=self.relu1(input)
        return input


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,k=1,bias=False):
        super(DepthwiseSeparableConv2d,self).__init__()

        self.depthwise=nn.Conv2d(in_channels=in_channels,
                                out_channels=in_channels*k,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=in_channels,
                                bias=bias)
        self.pointwise=nn.Conv2d(in_channels=in_channels*k,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=bias)
    def forward(self,input):
        input=self.depthwise(input)
        input=self.pointwise(input)
        return input

class DWSConvBnReluBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DWSConvBnReluBlock,self).__init__()

        self.dws_1=DepthwiseSeparableConv2d(in_channels=in_channels,out_channels=out_channels)
        self.bn_1=nn.BatchNorm2d(out_channels)

        self.rl=nn.ReLU(inplace=True)

        self.dws_2=DepthwiseSeparableConv2d(in_channels=out_channels,out_channels=out_channels)
        self.bn_2=nn.BatchNorm2d(out_channels)

    def forward(self,input):
        input=self.dws_1(input)
        input=self.bn_1(input)
        input=self.rl(input)

        input=self.dws_2(input)
        input=self.bn_2(input)
        input=self.rl(input)
        return input

class DWSConvBottleneck(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DWSConvBottleneck,self).__init__()

        self.dws_block_1=DWSConvBnReluBlock(in_channels=in_channels,
                                      out_channels=out_channels)
    def forward(self,input):
        input=self.dws_block_1(input)
        return input