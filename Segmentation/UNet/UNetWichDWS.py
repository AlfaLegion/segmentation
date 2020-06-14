from UNet.UnetBlocks import*
import torch
import torch.nn as nn

class UNetDWS(nn.Module):
    def __init__(self,input_channels,num_classes,num_filters):
        super(UNetDWS,self).__init__()


        print("net: UnetTransform")
        
        self.dws_1=DWSConvBnReluBlock(input_channels,num_filters)
        self.maxpool_1=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        self.dws_2=DWSConvBnReluBlock(num_filters,num_filters*2)
        self.maxpool_2=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        self.dws_3=DWSConvBnReluBlock(num_filters*2,num_filters*4)
        self.maxpool_3=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        self.dws_4=DWSConvBnReluBlock(num_filters*4,num_filters*8)
        self.maxpool_4=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)

        self.dws_bottleneck=DWSConvBottleneck(num_filters*8,num_filters*16)

        self.up_4=DeconvBnRelu(num_filters*16,num_filters*8,1,1)
        self.up_dws_4=DWSConvBnReluBlock(num_filters*16,num_filters*8)

        self.up_3=DeconvBnRelu(num_filters*8,num_filters*4,1,1)
        self.up_dws_3=DWSConvBnReluBlock(num_filters*8,num_filters*4)

        self.up_2=DeconvBnRelu(num_filters*4,num_filters*2,1,1)
        self.up_dws_2=DWSConvBnReluBlock(num_filters*4,num_filters*2)

        self.up_1=DeconvBnRelu(num_filters*2,num_filters,1,1)
        self.up_dws_1=DWSConvBnReluBlock(num_filters*2,num_filters)

        self.output=nn.Conv2d(num_filters,num_classes,kernel_size=1,stride=1,padding=0)

    def forward(self,input):
        down_1=self.dws_1(input)
        pool_1=self.maxpool_1(down_1)
         
        down_2=self.dws_2(pool_1)
        pool_2=self.maxpool_2(down_2)
         
        down_3=self.dws_3(pool_2)
        pool_3=self.maxpool_3(down_3)
         
        down_4=self.dws_4(pool_3)
        pool_4=self.maxpool_4(down_4)
              
        bottleneck=self.dws_bottleneck(pool_4)

        deconv_4=self.up_4(bottleneck)
        concat_4=torch.cat([deconv_4,down_4],dim=1)
        up_4=self.up_dws_4(concat_4)

        deconv_3=self.up_3(up_4)
        concat_3=torch.cat([deconv_3,down_3],dim=1)
        up_3=self.up_dws_3(concat_3)

        deconv_2=self.up_2(up_3)
        concat_2=torch.cat([deconv_2,down_2],dim=1)
        up_2=self.up_dws_2(concat_2)

        deconv_1=self.up_1(up_2)
        concat_1=torch.cat([deconv_1,down_1],dim=1)
        up_1=self.up_dws_1(concat_1)

        out=self.output(up_1)

        return out



if __name__=="__main__":
    model=UnetDWS(3,3,64).cpu()
    summary(model,(3, 256, 256),device="cpu")
    print(model)
