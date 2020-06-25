import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import time as t
import collections as cl
from torch.nn import functional as F
from torchvision import transforms, models
import torch
import albumentations as albu
from torchsummary import summary
#import matplotlib.pylab as plb
from UNet import Unet  
from UNet.UNetWichDWS import UNetDWS
from UNet import UNetTransferLerning  
from DataLoader import DataLoaderBacteriaBorderWatershed as DataLoader
import random
import math
def decode_segmap(image, nc=3):
   
  label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (0, 255, 0), (0, 0, 255), (255, 0, 0), (192, 128, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
 
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
   
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
     
  rgb = np.stack([r, g, b], axis=2)
  return rgb  

def instance_decode_map(mask):
    temp=np.expand_dims(mask, axis=2)
    contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print("count object: ",len(contours))

    avg_area=0.0
    for cntr in contours:
        avg_area+=cv2.contourArea(cntr)

    avg_area/=len(contours)
    print("avg area: ",avg_area)
    _n_clusters=len(contours)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]
    random.shuffle(colors)
    instance_color_map=np.zeros((temp.shape[0],temp.shape[1],3),np.uint8)

    for i in range(_n_clusters):
        color=(np.array(colors[i][:3]) * 255).astype(np.uint8)
        color =tuple ([int (color[x]) for x in range(3)])
        cv2.drawContours(instance_color_map,contours,i,color,-1,hierarchy=hierarchy)
        #cv2.imshow("instance_color_map",instance_color_map)
        #cv2.waitKey()

    instance_color_map=cv2.cvtColor(instance_color_map,cv2.COLOR_RGB2BGR)
    return instance_color_map


def SegmentationEval3ClassesWithBorder(pathModel):
    img_size=(256,256)
    device=torch.device("cuda")
    #pathDataInput=r'D:\datasets\data-science-bowl-2018\transform_test'
    #pathDataInput=r'D:\datasets\HELA\test\DIC-C2DH-HeLa\DIC-C2DH-HeLa\01'
    #pathDataInput=r'D:\datasets\Fluo\train\Fluo-N2DH-GOWT1\Fluo-N2DH-GOWT1\01'
    pathDataInput=r'D:\datasets\11_06_2019_wtn\deb'
    model=torch.jit.load(pathModel,device)

    listFiles=os.listdir(pathDataInput)

    #rdCrop=albu.Resize(img_size[0],img_size[1])
    rdCrop=albu.RandomCrop(256,256)
    cnt=0
    for file in listFiles:
        imgPath=os.path.join(pathDataInput,file)
       
        #imgName=os.listdir(imgPath)
        #imgName=os.path.join(pathDataInput,file,"images",imgName[0])
        img=cv2.imread(imgPath,cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        # new dataset
        img=img[0:725,:,:]
        #img=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
        #data = {"image": img, "mask": np.empty(img.shape), "whatever_data": "asd", "additional": "hello"}
        #cloneMat=rdCrop(image=img)["image"]

        ###
        #img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        #img=cv2.bitwise_not(img)
        #img=cv2.resize(img,(512,512))
        #cloneMat=np.copy(img[0:img_size[0],0:img_size[1]])
        #cloneMat=cv2.resize(img,(256,256))

        #if cloneMat.shape[0]!=img_size[0] or cloneMat.shape[1]!=img_size[1]:
        cloneMat = cv2.resize(img,(256,256),None,0,0,cv2.INTER_AREA)

        input=transforms.functional.to_tensor(cloneMat)
        input = transforms.functional.normalize(input, mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        input.unsqueeze_(0)
        input = input.to(device)
        out = model(input)
        #mask = out.detach().cpu().numpy()[0]
        mask=out.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)

        colorSegMap=decode_segmap(mask)

        cloneMat=cv2.cvtColor(cloneMat,cv2.COLOR_RGB2BGR)

        #cloneMat=cv2.resize(cloneMat,(img.shape[1],img.shape[0]))
        colorSegMap=cv2.resize(colorSegMap,(round(img.shape[1]/4),round(img.shape[0]/4)))
        img=cv2.resize(img,(round(img.shape[1]/4),round(img.shape[0]/4)))

        cv2.imshow("colorSegMap",colorSegMap)
        cv2.imshow("img",img)

        cv2.imwrite('./bin/'+str(cnt)+"_bin.png",colorSegMap)
        cv2.imwrite('./img/'+str(cnt)+"_im.png",img)
        #cnt+=
        print('\n')
        cv2.waitKey()

def InstanceSegmentationEval(pathModel,pathDataInput):
    img_size=(256,256)
    device=torch.device("cuda")


    model=torch.jit.load(pathModel,device)

    listFiles=os.listdir(pathDataInput)

    #rdCrop=albu.Resize(img_size[0],img_size[1])
    rdCrop=albu.RandomCrop(256,256)
    cnt=0

    for file in listFiles:
        imgPath=os.path.join(pathDataInput,file) 
        print(imgPath)
        img=cv2.imread(imgPath,cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        all_mask=np.zeros((img.shape[0],img.shape[1]),np.uint8)
        
        cnt_r=math.floor(img.shape[0]/img_size[0])
        cnt_c= math.floor(img.shape[1]/img_size[1])
        for i in range(cnt_r):
            for j in range(cnt_c):
                roi_img=img[i*img_size[0]:(i+1)*img_size[0],j*img_size[1]:(j+1)*img_size[1]]
                input=transforms.functional.to_tensor(roi_img)
                input = transforms.functional.normalize(input, mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
                input.unsqueeze_(0)
                input = input.to(device)
                out = model(input)
                mask=out.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)
                np.copyto(all_mask[i*img_size[0]:(i+1)*img_size[0],j*img_size[1]:(j+1)*img_size[1]],mask)
                colorSegMap=decode_segmap(mask)
                #cv2.imshow("roi_img",roi_img)
                #cv2.imshow("colorSegMap",colorSegMap)
                #cv2.waitKey()


        img=img[0:725,:]
        all_mask=all_mask[0:725,:]     
        #processing segmentation map
        all_mask=(all_mask==1).astype(np.uint8)*255
        cv2.morphologyEx(all_mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),all_mask)
        instance_color_map=instance_decode_map(all_mask)
        #colorSegMap=decode_segmap(all_mask,2)

        #instance_color_map=cv2.resize(instance_color_map,(round(img.shape[1]/4),round(img.shape[0]/4)))
        #all_mask=cv2.resize(all_mask,(round(img.shape[1]/4),round(img.shape[0]/4)))
        #img=cv2.resize(img,(round(img.shape[1]/4),round(img.shape[0]/4)))

        cv2.imshow("mask",all_mask)
        cv2.imshow("instance_color_map",instance_color_map)
        cv2.imshow("img",img)

        #cv2.imwrite('./bin_vgg_adabatch/'+"vgg_adabt_"+str(cnt)+"_bin.png",instance_color_map)
        #cv2.imwrite('./img/'+"static6_"+str(cnt)+"_img.png",img)
        print('./img/'+"static6"+str(cnt)+"_img.png")
        cnt+=1
        print('\n')
        cv2.waitKey()




if __name__=="__main__":
   
    #InstanceSegmentationEval("D:/Projects/Segmentation models/TheBestModels/d2/adabatchvgg19/AB_unet_vgg_2.pt",r'D:\datasets\11_06_2019_wtn\deb3')
    InstanceSegmentationEval("D:/Projects/Segmentation models/TheBestModels/d2/UNetVGG19BN/UNetVGG_0.770019063711179_.pt",r'D:\datasets\11_06_2019_wtn\deb2')
   



