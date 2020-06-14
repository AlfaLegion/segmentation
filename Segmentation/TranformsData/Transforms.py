import torch
import cv2
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose
import collections
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        #assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

class ToLabel(object):
    def __call__(self, inputs):
        
        
        _,numpyTen=cv2.threshold(inputs,40,1,cv2.THRESH_BINARY)
        tensor= torch.from_numpy(numpyTen)
        tensor.unsqueeze_(0)

        return tensor.float()

class ToLabelMultyClasses(object):
    def __call__(self,inputs):

        tensor= torch.from_numpy(inputs)
        #tensor.unsqueeze_(0)
        return tensor.long()

class CreateInputTensorRGBD(object):
    def __init__(self):
        self.toTensor=transforms.ToTensor()
    def __call__(self,img,depth):
        img_chw=self.toTensor(img)
        dp=self.toTensor(depth)
        input_net=torch.cat((img_chw,dp),dim=0)

        return input_net


class RandomCrop(object):
    def __init__(self,sizeImg):
        self.size=sizeImg

    def __call__(self,input,label):
        dr=input.shape[0]-self.size[0]
        dc=input.shape[1]-self.size[1]
        if dc>0:
            dc=torch.randint(0,dc+1,(1,)).item()
        if dr>0:
            dr=torch.randint(0,dr+1,(1,)).item()

        transform_input=input[dr:dr+self.size[0],dc:dc+self.size[1]]
        transform_mask=label[dr:dr+self.size[0],dc:dc+self.size[1]]
        return transform_input,transform_mask


class ReLabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        # assert isinstance(input, torch.LongTensor), 'tensor needs to be LongTensor'
        for i in inputs:
            i[i == self.olabel] = self.nlabel
        return inputs

class TransformNoisyNormal(object):
    def __init__(self,mean,sigma,alpha,betta):
        self.mean = mean
        self.sigma = sigma
        self.betta = betta
        self.alpha = alpha
    def __call__(self,input):

        probVal=torch.randint(0,2,(1,)).item()
        if probVal==0:
            noisy = np.empty(input.shape,dtype=np.float)
            cv2.randn(noisy,self.mean,self.sigma)
            output = cv2.addWeighted(input,self.alpha,noisy,self.betta,0,None,cv2.CV_8UC3)
        else:
            output=input
        return output

class TransformFlip(object):
    def __call__(self,input,target,depth=None):

        probFlip=torch.randint(0,5,(1,)).item()
        if probFlip == 0:
            output = cv2.flip(input,0)
            outputTraget=cv2.flip(target,0)
        elif probFlip == 1:
            output = cv2.flip(input,1)
            outputTraget=cv2.flip(target,1)
        elif probFlip == 2:
            output = cv2.flip(input,-1)
            outputTraget=cv2.flip(target,-1)
        elif probFlip >= 3:
            output=input
            outputTraget=target

        return output,outputTraget


class TransformBlur(object):
    def __init__(self,sz_filt):
        self.sz_filt=sz_filt
    def __call__(self,input):
        probBlur=torch.randint(0,4,(1,)).item()
        #probCount=torch.randint(1,3,(1,)).item()

        if probBlur==0:
           output= cv2.blur(input,(self.sz_filt,self.sz_filt))
        elif probBlur==1:
            output=cv2.medianBlur(input,self.sz_filt)
        elif probBlur==2:
            output=cv2.GaussianBlur(input,(self.sz_filt,self.sz_filt),0)
        else:
            output=input
        return output

class TransformColorChannels(object):
    def __call__(self,input):
        probCvt=torch.randint(0,3,(3,))

        
        splitImgs=cv2.split(input)

        output=cv2.merge([splitImgs[probCvt[0].item()]
                         ,splitImgs[probCvt[1].item()],
                         splitImgs[probCvt[2].item()]])

        return output


class TransformsColorJitter(object):
    def __init__(self,brightness=0,contrast=0,saturation=0,hue=0 ):
        self.colorJitter=transforms.ColorJitter(brightness ,contrast,saturation,hue )
        
    def __call__(self,input):
        pilImg=transforms.functional.to_pil_image(input)
        pilImg=self.colorJitter(pilImg)
        out=np.array(pilImg)
        return out


class ToSP(object):
    def __init__(self, size):
        self.scale2 = Scale(size/2, Image.NEAREST)
        self.scale4 = Scale(size/4, Image.NEAREST)
        self.scale8 = Scale(size/8, Image.NEAREST)
        self.scale16 = Scale(size/16, Image.NEAREST)
        self.scale32 = Scale(size/32, Image.NEAREST)

    def __call__(self, input):
        input2 = self.scale2(input)
        input4 = self.scale4(input)
        input8 = self.scale8(input)
        input16 = self.scale16(input)
        input32 = self.scale32(input)
        inputs = [input, input2, input4, input8, input16, input32]
        # inputs = [input]

        return inputs