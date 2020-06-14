import torch
import torch.nn as nn
from torchsummary import summary
from UNet.UnetBlocks import*

class UNet(nn.Module):
    def __init__(self,input_dim,out_dim,start_channels):
        super(UNet,self).__init__()

        print("net: Unet")

        self.out_dim = out_dim

        self.downBlock_1 = ConvBlock(input_dim,start_channels,1)
        
        self.downBlock_2 = ConvBlock(start_channels,start_channels * 2,1)        

        self.downBlock_3 = ConvBlock(start_channels * 2,start_channels * 4,1)

        self.downBlock_4 = ConvBlock(start_channels * 4,start_channels * 8,1)
       

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
        #self.act_sigmoid_out = nn.Sigmoid()

    def forward(self,input):

        down_1 = self.downBlock_1(input)
        pool_1 = self.mp(down_1)

        down_2 = self.downBlock_2(pool_1)
        pool_2 = self.mp(down_2)

        down_3 = self.downBlock_3(pool_2)
        pool_3 = self.mp(down_3)

        down_4 = self.downBlock_4(pool_3)
        pool_4 = self.mp(down_4)

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

        #if self.out_dim > 1:
        #    output = F.log_softmax(output,dim=1)

        return output


if __name__=="__main__":
    model=UNet(3,3,64).cpu()
    summary(model,(3, 512, 512),device='cpu')
    print(model)










