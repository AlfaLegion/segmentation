import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
class DepthwiseConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ksize,stride,padding):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=ksize,
                                 stride=stride,
                                 padding=padding,
                                 groups=in_channels,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        input = self.depthwise(input)
        input = self.bn(input)
        input = self.relu(input)
        return input

class PointwiseConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PointwiseConvBlock,self).__init__()
        self.pointwise = nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=(1,1),
                                 stride=1,
                                 padding=0,bias=False)
        self.bn=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()

    def forward(self,input):
        input = self.pointwise(input)
        input = self.bn(input)
        input = self.relu(input)
        return input

class DSConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(DSConv2d,self).__init__()

        self.DWConv2d=DepthwiseConvBlock(in_channels,in_channels,(3,3),stride,1)
        self.PWConv2d=PointwiseConvBlock(in_channels,out_channels)

    def forward(self,input):
        input=self.DWConv2d(input)
        input=self.PWConv2d(input)
        return input


class LinearBottleneck(nn.Module):

    def __init__(self,in_channels,out_chnannels,stride,t):
        super(LinearBottleneck,self).__init__()

        self.pw1=PointwiseConvBlock(in_channels,in_channels*t)
        self.dw=DepthwiseConvBlock(in_channels*t,in_channels*t,(3,3),stride,1)
        self.pw2=PointwiseConvBlock(in_channels*t,out_chnannels)
        self.use_res_connect=(stride==1 and in_channels==out_chnannels)

    def forward(self,input):
        x=self.pw1(input)
        x=self.dw(x)        
        x=self.pw2(x)
        if self.use_res_connect:
            x=x+input
        return x


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h,w), mode='bilinear',
                                align_corners=True) for stage in self.stages] + [feats]
        # import pdb;pdb.set_trace()
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


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

class LearningToDownSample(nn.Module):
    def __init__(self,input_channels,ds_channels1=32,ds_channels2=48,ds_channels3=64):
        super(LearningToDownSample,self).__init__()

        self.Conv2dBnReLU=nn.Sequential(nn.Conv2d(in_channels=input_channels,out_channels=ds_channels1,kernel_size=(3,3),stride=2,padding=1),
                                        nn.BatchNorm2d(ds_channels1),
                                        nn.ReLU())
        self.dsConv2d_1=DSConv2d(ds_channels1,ds_channels2,2)
        self.dsConv2d_2=DSConv2d(ds_channels2,ds_channels3,2)

    def forward(self,input):
        x =self.Conv2dBnReLU(input)
        x =self.dsConv2d_1(x)
        x=self.dsConv2d_2(x)
        return x

class GlobalFeatureExtractor(nn.Module):
    def __init__(self,out_channels_block,out_channels,t):
        super(GlobalFeatureExtractor,self).__init__()

        self.bottleneck1=nn.Sequential(
            LinearBottleneck(out_channels_block[0],out_channels_block[0],2,6),
            LinearBottleneck(out_channels_block[0],out_channels_block[0],1,6),
            LinearBottleneck(out_channels_block[0],out_channels_block[0],1,6)

            )

        self.bottleneck2=nn.Sequential(
                 LinearBottleneck(out_channels_block[0],out_channels_block[1],2,6),
                 LinearBottleneck(out_channels_block[1],out_channels_block[1],1,6),
                 LinearBottleneck(out_channels_block[1],out_channels_block[1],1,6)
            )

        self.bottleneck3=nn.Sequential(
                 LinearBottleneck(out_channels_block[1],out_channels_block[2],1,6),
                 LinearBottleneck(out_channels_block[2],out_channels_block[2],1,6),
                 LinearBottleneck(out_channels_block[2],out_channels_block[2],1,6)
            )

        self.ppm=PSPModule(out_channels_block[2],out_channels)

    def forward(self,input):
        input=self.bottleneck1(input)
        input=self.bottleneck2(input)
        input=self.bottleneck3(input)
        input=self.ppm(input)
        return input

class FeatureFusionModule(nn.Module):
    def __init__ (self,upper_in_channels,lower_in_channels,out_channels):
        super(FeatureFusionModule,self).__init__()

        self.DWConvDilation=nn.Conv2d(in_channels=upper_in_channels,out_channels=upper_in_channels,kernel_size=(3,3),stride=1,padding=1,dilation=1,groups=upper_in_channels)
        self.conv_upper=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(1,1),stride=1)

        self.conv_lower=nn.Conv2d(in_channels=lower_in_channels,out_channels=out_channels,kernel_size=(1,1),stride=1)

    def forward(self,upper_input,lower_input):
        upper_input=F.interpolate(upper_input,scale_factor=4,mode='bilinear',align_corners=True)
        upper_input=self.DWConvDilation(upper_input)
        upper_input=self.conv_upper(upper_input)

        lower_input=self.conv_lower(lower_input)

        out=torch.add(upper_input,lower_input)
        return out

class Classifier(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Classifier,self).__init__()

        self.DSconv1=DSConv2d(in_channels,in_channels,1)
        self.DSconv2=DSConv2d(in_channels,in_channels,1)
        self.PWconv1=PointwiseConvBlock(in_channels,in_channels)
        self.deconv=DeconvBnRelu(in_channels,out_channels,1,1)

    def forward(self, input):
        input=self.DSconv1(input)
        input=self.DSconv2(input)
        input=self.PWconv1(input)
        input= self.deconv(input)
        return input


class Fast_SCNN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Fast_SCNN,self).__init__()
        self.LearningToDownSample=LearningToDownSample(in_channels)
        self.GlobalFeatureExtractor=GlobalFeatureExtractor([64,96,128],128,6)
        self.FeatureFusionModule=FeatureFusionModule(128,64,128)
        self.Classifier=Classifier(128,out_channels)

    def forward(self,input):
        out=self.LearningToDownSample(input)
        x=self.GlobalFeatureExtractor(out)
        out=self.FeatureFusionModule(x,out)
        out=self.Classifier(out)
        return out
   
if __name__=="__main__":


   testModule=Fast_SCNN(3,23)
   
   input=torch.rand(1,3,1024,2048)

   out=testModule(input)

   print(out.shape)
    