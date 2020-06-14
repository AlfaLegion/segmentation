import torch
import cv2
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import albumentations as albu


class DataLoaderBacteriaBorderWatershed(torch.utils.data.Dataset):
    def __init__(self,dataRoot,transform=None):
        self.dataRoot=dataRoot
        self.transform=transform
        listFiles=os.listdir(self.dataRoot)
        self.dataset=list()
        self.ToTensor=transforms.ToTensor()
        self.Normalize=transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        for file in listFiles:
            pathImg=os.path.join(self.dataRoot,file,"images")
            pathBorder=os.path.join(self.dataRoot,file,"border")
            
            listImgs=os.listdir(pathImg)
            listBorders=os.listdir(pathBorder)

            lsMask=list()
            fileMasks=os.listdir(os.path.join(self.dataRoot,file,"masks"))
            for fileMask in fileMasks:
                lsMask.append(os.path.join(self.dataRoot,file,"masks",fileMask))
            self.dataset.append({
                "image":os.path.join(pathImg,listImgs[0]),
                "border":os.path.join(pathBorder,listBorders[0]),
                "labels":lsMask
                })
        print("Number of label: ",len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        datagramm=self.dataset[index]

        input=cv2.imread(datagramm["image"],cv2.IMREAD_COLOR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        border=cv2.imread(datagramm["border"],cv2.IMREAD_GRAYSCALE)
        #D:\datasets\data-science-bowl-2018\wich_border\stage1_train\00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552\masks
        generalMask=np.zeros_like(border)
        for maskName in datagramm["labels"]:
            mask=cv2.imread(maskName,cv2.IMREAD_GRAYSCALE)
            generalMask=cv2.bitwise_or(generalMask,mask)

        _,generalMask=cv2.threshold(generalMask,50,1,cv2.THRESH_BINARY)
        _,border=cv2.threshold(border,50,2,cv2.THRESH_BINARY)
        generalMask=cv2.bitwise_or(generalMask,border)

        _,generalMask=cv2.threshold(generalMask,2,2,cv2.THRESH_TRUNC)

        #print(np.unique(generalMask))
        #generalMask=generalMask*80
        #0 - background
        #1 - objects
        #2 - borders

        if  self.transform:
            augmented=self.transform(image=input,mask=generalMask)
        else:
            augmented={"image":input,"mask":generalMask}
        
        input_tensor=self.ToTensor(augmented["image"])
        input_tensor=self.Normalize(input_tensor)

        label_tensor=torch.from_numpy(augmented["mask"])


        #cv2.imshow("input",augmented["image"])
        ##cv2.imshow("border",border*)
        #cv2.imshow("generalMask",augmented["mask"]*100)
        #cv2.waitKey()
        return input_tensor, label_tensor









if __name__=="__main__":
    img_size=(256,256)
    transform=albu.Compose([
         #albu.RandomCrop(img_size[0],img_size[1]),
         albu.RandomResizedCrop(img_size[0],img_size[1],interpolation=cv2.INTER_NEAREST,ratio=(1,1.2)),
         

        #albu.Rotate(interpolation=cv2.INTER_NEAREST),
        #albu.HorizontalFlip(),
        #albu.ElasticTransform(alpha_affine=10,interpolation=cv2.INTER_NEAREST),
        #albu.OpticalDistortion(interpolation=cv2.INTER_NEAREST),

        #pixel transform
        albu.RandomGamma((70,150),p=1),
         albu.Blur(5),
         albu.RandomBrightness(limit=(0.1,0.2)),
         albu.RandomContrast(limit=(0,0.2),p=1),
         albu.MotionBlur( blur_limit =7),
        ])
    dataset=DataLoaderBacteriaBorderWatershed(r'D:\datasets\data-science-bowl-2018\wich_border\stage1_train',transform=transform)
    #data=dataset[2]
    
    for i in range(670):
        input,label=dataset[i]
        cond_1 = label>=3 
        res=np.unique(label)>=3
        print(np.sum(res))


