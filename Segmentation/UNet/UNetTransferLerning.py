import torch
import torchvision.models as models
from torchsummary import summary
import torch.nn as nn
from UNet.UnetBlocks import*



class UNetVGG(nn.Module):
    def __init__(self,input_dim,out_dim,start_channels):
        super(UNetVGG,self).__init__()

        print("net: UNetVGG19_BN")

        self.out_dim=out_dim
        vgg=models.vgg19_bn(True)

        ##encoder
        self.downBlock_1=nn.Sequential(
            vgg.features[0:6]
            )

        self.downBlock_2=nn.Sequential(
            vgg.features[7:13]
            )

        self.downBlock_3=nn.Sequential(
            vgg.features[14:26]
            )
        self.downBlock_4=nn.Sequential(
            vgg.features[27:39]
            )
        #self.encoder=nn.Sequential(vgg.features[0:40])

        ##

        self.bottleneck = Bottleneck(start_channels * 8,start_channels * 16,1)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.deconvBlock_1 = DeconvBnRelu(start_channels * 16,start_channels * 8,1,1)
        self.convBlock_1 = ConvBlock(start_channels * 16,start_channels * 8,1)
       

        self.deconvBlock_2 = DeconvBnRelu(start_channels * 8,start_channels * 4,1,1)
        self.convBlock_2 = ConvBlock(start_channels * 8,start_channels * 4,1)


        self.deconvBlock_3 = DeconvBnRelu(start_channels * 4,start_channels * 2,1,1)
        self.convBlock_3 = ConvBlock(start_channels * 4,start_channels * 2,1)

        self.deconvBlock_4 = DeconvBnRelu(start_channels * 2,start_channels,1,1)
        self.convBlock_4 = ConvBlock(start_channels * 2,start_channels,1)

        self.outConv = nn.Conv2d(start_channels,out_dim,1,stride=1,padding=0)


    def forward(self,input):

        ##encoder
        down_1 = self.downBlock_1(input)
        pool_1 = self.mp(down_1)

        down_2 = self.downBlock_2(pool_1)
        pool_2 = self.mp(down_2)

        down_3 = self.downBlock_3(pool_2)
        pool_3 = self.mp(down_3)

        down_4 = self.downBlock_4(pool_3)
        pool_4 = self.mp(down_4)

        ###

        bottleneck = self.bottleneck(pool_4)

        decon_1 = self.deconvBlock_1(bottleneck)
        concat_1 = torch.cat([decon_1,down_4],dim=1)
        up_1 = self.convBlock_1(concat_1)
        

        decon_2 = self.deconvBlock_2(up_1)
        concat_2 = torch.cat([decon_2,down_3],dim =1)
        up_2 = self.convBlock_2(concat_2)

        decon_3 = self.deconvBlock_3(up_2)
        concat_3 = torch.cat([decon_3,down_2],dim =1)
        up_3 = self.convBlock_3(concat_3)

        decon_4 = self.deconvBlock_4(up_3)
        concat_4 = torch.cat([decon_4,down_1],dim =1)
        up_4 = self.convBlock_4(concat_4)

        output = self.outConv(up_4)
        return output


class UNetResNet(nn.Module):
    def __init__(self,input_dim,out_dim,start_channels,type_base_model='resnet18'):
        super(UNetResNet,self).__init__()

        self.out_dim=out_dim

        if type_base_model=='resnet18':
            ResnetBaseModel=models.resnet18(True)
        elif type_base_model=='resnet34':
            ResnetBaseModel=models.resnet34(True)
        else:
            raise Exception("invalid type of base model") 
        ResnetBaseModel.cpu()

        print('net: UNet_'+type_base_model)

        self.start_conv=nn.Sequential(nn.Conv2d(
            in_channels=input_dim,out_channels=start_channels,kernel_size=3,stride=1,padding=1),
                                      nn.ReLU(True))

        self.downBlock_1=nn.Sequential(ResnetBaseModel.layer1)
        self.downBlock_2=nn.Sequential(ResnetBaseModel.layer2)
        self.downBlock_3=nn.Sequential(ResnetBaseModel.layer3)
        self.downBlock_4=nn.Sequential(ResnetBaseModel.layer4)

        isFreeze=False
        if isFreeze:
            for param in self.downBlock_1.parameters():
                param.requires_grad = False 
            for param in self.downBlock_2.parameters():
                param.requires_grad = False 
            for param in self.downBlock_3.parameters():
                param.requires_grad = False 
            for param in self.downBlock_4.parameters():
                param.requires_grad = False 




        self.bottleneck = Bottleneck(start_channels * 8,start_channels * 16,1)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        self.deconvBlock_1 = DeconvBnRelu(start_channels * 16,start_channels * 8,1,1)
        self.convBlock_1 = ConvBlock(start_channels * 16,start_channels * 8,1)
       

        self.deconvBlock_2 = DeconvBnRelu(start_channels * 8,start_channels * 4,1,1)
        self.convBlock_2 = ConvBlock(start_channels * 8,start_channels * 4,1)


        self.deconvBlock_3 = DeconvBnRelu(start_channels * 4,start_channels * 2,1,1)
        self.convBlock_3 = ConvBlock(start_channels * 4,start_channels * 2,1)

        self.deconvBlock_4 = DeconvBnRelu(start_channels * 2,start_channels,1,1)
        self.convBlock_4 = ConvBlock(start_channels * 2,start_channels,1)

        self.outConv = nn.Conv2d(start_channels,out_dim,1,stride=1,padding=0)

    def forward(self,input):

        down_1=self.start_conv(input)
        down_1=self.downBlock_1(down_1) #64

        down_2 = self.downBlock_2(down_1) #128

        down_3 = self.downBlock_3(down_2) #256

        down_4 = self.downBlock_4(down_3) #512

        pool_bottleneck=self.mp(down_4)
        bottleneck = self.bottleneck(pool_bottleneck) #1024

        decon_1 = self.deconvBlock_1(bottleneck)
        concat_1 = torch.cat([decon_1,down_4],dim=1)
        up_1 = self.convBlock_1(concat_1)

        decon_2 = self.deconvBlock_2(up_1)
        concat_2 = torch.cat([decon_2,down_3],dim =1)
        up_2 = self.convBlock_2(concat_2)

        decon_3 = self.deconvBlock_3(up_2)
        concat_3 = torch.cat([decon_3,down_2],dim =1)
        up_3 = self.convBlock_3(concat_3)

        decon_4 = self.deconvBlock_4(up_3)
        concat_4 = torch.cat([decon_4,down_1],dim =1)
        up_4 = self.convBlock_4(concat_4)

        output = self.outConv(up_4)
        return output

if __name__ == "__main__":
    model=UNetResNet(3,3,64).cuda()
    summary(model,(3,256,256),device='cuda')
    print(model)

    

    
