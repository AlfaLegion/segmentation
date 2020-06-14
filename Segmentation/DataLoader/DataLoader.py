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
import Transforms as trs
##########################class dataset###############
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir,img_transofrm=None,label_transform=None,nameImageFile="images",nameMasksFile="masks"):
        self.root_dir = root_dir
        self.img_transofrm = img_transofrm
        self.label_transform = label_transform
        self.files = list()

        ####Augmentation###
        self.TrChannels=TransformColorChannels()
        self.TrFlip=TransformFlip()
        self.colorJitter=TransformsColorJitter(0.65,0.8,0.8,0.09)
        self.TrNoisy=TransformNoisyNormal((128,128,128),(128,128,128),0.98,0.02)
        self.TrBlur=TransformBlur(3)
        ###################

        listNamesClass = os.listdir(root_dir)
        print("Found files: ",listNamesClass)
        inex = 0



        for nameClass in listNamesClass:
            pathClass = os.path.join(root_dir,nameClass)
            pathImage = os.path.join(pathClass,nameImageFile)
            pathMask = os.path.join(pathClass,nameMasksFile)

            listImage = os.listdir(pathImage)
            listMask = os.listdir(pathMask)
            lenIms = len(listImage)
            lenMsks = len(listMask)
            if lenIms != lenMsks:
                strError = "error of the dimension of the list of images (" + str(lenIms) + ") and masks (" + str(lenMsks) + ")"
                raise IOError(strError)

            for i in range(lenIms):
                img_file = os.path.join(pathImage,listImage[i])
                msk_file = os.path.join(pathMask,listMask[i])
                self.files.append({
                    "image":img_file,
                    "label":msk_file
                    })

        print("Number of labels: ",len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        
        datafile = self.files[index]

        img_file = datafile["image"]
        img=cv2.imread(img_file,cv2.IMREAD_COLOR)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        

        label_file = datafile["label"]
        label=cv2.imread(label_file,cv2.IMREAD_GRAYSCALE)



        ####Augmentation###
       
        #img=self.TrChannels(img)
        img,label=self.TrFlip(img,label)
        img=self.colorJitter(img)
        #img=self.TrBlur(img)
        img=self.TrNoisy(img)
        ###################



        #cv2.imshow("image",img)
        #cv2.imshow("label",label)
        #cv2.waitKey()

        #label = Image.open(label_file)
        #label = np.array(label)

        if self.img_transofrm is not None:
            img_o = self.img_transofrm(img)
            imgs = img_o
        else:
            imgs = img

        if self.label_transform is not None:
            label_o = self.label_transform(label)
            labels = label_o
        else:
            labels = label

        
        
        return imgs, labels


class SegmentationDatasetAnyMaskFilesBinaryClass(torch.utils.data.Dataset):
    def __init__(self, root_dir,shape_data,img_transofrm=None,label_transform=None,nameImageFile="images",nameMasksFile="masks"):

        self.shape_data=shape_data
        self.root_dir=root_dir
        self.img_transofrm = img_transofrm
        self.label_transform = label_transform
        self.nameImageFile=nameImageFile
        self.nameMasksFile=nameMasksFile
        self.files = list()

        self.files=os.listdir(self.root_dir)
        print("Found files: ",len(self.files))
    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):

        changeFile=self.files[index]

        pathImage=os.path.join(self.root_dir,changeFile,self.nameImageFile)
        pathMasks=os.path.join(self.root_dir,changeFile,self.nameMasksFile)

        listImage=os.listdir(pathImage)
        listMasks=os.listdir(pathMasks)

        #read input
        input_img=cv2.imread(os.path.join(pathImage,listImage[0]),cv2.IMREAD_COLOR)
        
        if input_img.shape[:2]!=self.shape_data[:2]:
            input_img=cv2.resize(input_img,self.shape_data[:2])
            

        #read label
        listLabelsNp=[]
        labels=np.zeros(self.shape_data[:2],np.uint8)
        for namelabel in listMasks:
            label=cv2.imread(os.path.join(pathMasks,namelabel),cv2.IMREAD_GRAYSCALE)

            if label.shape!=self.shape_data[:2]:
                label=cv2.resize(label,self.shape_data[:2])

            labels= cv2.bitwise_or(labels,label)
            listLabelsNp.append(label)

        _,labels=cv2.threshold(labels,40,1,cv2.THRESH_BINARY)
        #build weight map
        weight_map=make_weight_map(np.array( listLabelsNp),10,5)
        weight_map_tensor=torch.from_numpy(weight_map)           

        if self.img_transofrm is not None:
            input_tensor = self.img_transofrm(input_img)
        else:
            input_tensor = input_img

        if self.label_transform is not None:
            labels_tensor = self.label_transform(labels)
        else:
           labels_tensor=labels


        return input_tensor, labels_tensor,weight_map_tensor

class SegmentationDatasetAnyMaskFilesBinaryClassNotMap(torch.utils.data.Dataset):
    def __init__(self, root_dir,shape_data,img_transofrm=None,label_transform=None,nameImageFile="images",nameMasksFile="masks"):

        self.shape_data=shape_data
        self.root_dir=root_dir
        self.img_transofrm = img_transofrm
        self.label_transform = label_transform
        self.nameImageFile=nameImageFile
        self.nameMasksFile=nameMasksFile
        self.files = list()
        self.files=os.listdir(self.root_dir)


         ####Augmentation###
        self.TrChannels=trs.TransformColorChannels()
        self.TrFlip=trs.TransformFlip()
        self.colorJitter=trs.TransformsColorJitter(0.65,0.8,0.8,0.09)
        self.TrNoisy=trs.TransformNoisyNormal((128,128,128),(128,128,128),0.98,0.02)
        self.TrBlur=trs.TransformBlur(3)
        ###################


        print("Found files: ",len(self.files))
    def __len__(self):
        return len(self.files)

    def __getitem__(self,index):

        changeFile=self.files[index]

        pathImage=os.path.join(self.root_dir,changeFile,self.nameImageFile)
        pathMasks=os.path.join(self.root_dir,changeFile,self.nameMasksFile)

        listImage=os.listdir(pathImage)
        listMasks=os.listdir(pathMasks)

        #read input
        input_img=cv2.imread(os.path.join(pathImage,listImage[0]),cv2.IMREAD_COLOR)
        
        if input_img.shape[:2]!=self.shape_data[:2]:
            input_img=cv2.resize(input_img,self.shape_data[:2])
            

        #read label
        listLabelsNp=[]
        labels=np.zeros(self.shape_data[:2],np.uint8)
        for namelabel in listMasks:
            label=cv2.imread(os.path.join(pathMasks,namelabel),cv2.IMREAD_GRAYSCALE)

            if label.shape!=self.shape_data[:2]:
                label=cv2.resize(label,self.shape_data[:2])

            labels= cv2.bitwise_or(labels,label)
            listLabelsNp.append(label)

        #_,labels=cv2.threshold(labels,40,1,cv2.THRESH_BINARY)

         ####Augmentation###
       
        #img=self.TrChannels(img)
        input_img,labels=self.TrFlip(input_img,labels)
        input_img=self.colorJitter(input_img)
        #img=self.TrBlur(img)
        input_img=self.TrNoisy(input_img)
        ###################


        #build weight map
        #weight_map=make_weight_map(np.array( listLabelsNp),10,5)
        #weight_map_tensor=torch.from_numpy(weight_map)           

        if self.img_transofrm is not None:
            input_tensor = self.img_transofrm(input_img)
        else:
            input_tensor = input_img

        if self.label_transform is not None:
            labels_tensor = self.label_transform(labels)
        else:
           labels_tensor=labels


        return input_tensor, labels_tensor

class DatasetInstrumental(torch.utils.data.Dataset):
    def __init__ (self,rootData,shapeData,imgTransforms=None, labelTransforms=None):
        self.rootData=rootData
        self.shapeData=shapeData
        self.imgTransforms=imgTransforms
        self.labelTransforms=labelTransforms
        self.filesDataset=list()

         ####Augmentation###
        self.TrChannels=trs.TransformColorChannels()
        self.TrFlip=trs.TransformFlip()
        self.colorJitter=trs.TransformsColorJitter(0.65,0.8,0.8,0.09)
        self.TrNoisy=trs.TransformNoisyNormal((128,128,128),(128,128,128),0.9,0.1)
        self.TrBlur=trs.TransformBlur(5)
        ###################

        rootFiles1=os.listdir(rootData)
        for fileName in rootFiles1:
            cropped_train=os.path.join(rootData,fileName,"cropped_train")
            instrument_datasets=os.listdir(cropped_train)
            for instrument_dataset in instrument_datasets:
                imagesPath=os.path.join(cropped_train,instrument_dataset,"images")
                masksPath=os.path.join(cropped_train,instrument_dataset,"binary_masks")

                images=os.listdir(imagesPath)
                masks=os.listdir(masksPath)

                lenImgs=len(images)
                lenMasks=len(masks)
                if lenImgs!=lenMasks:
                    strError = "error of the dimension of the list of images (" + str(lenIms) + ") and masks (" + str(lenMsks) + ")"
                    raise IOError(strError)
                for i in range(lenImgs):
                    img_file = os.path.join(imagesPath,images[i])
                    msk_file = os.path.join(masksPath,masks[i])
                    self.filesDataset.append({
                        "image":img_file,
                        "label":msk_file
                        })

    def __len__(self):
        return len(self.filesDataset)

    def __getitem__(self, index):
        dataGramm = self.filesDataset[index]

        img=cv2.imread(dataGramm["image"],cv2.IMREAD_COLOR)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        label=cv2.imread(dataGramm["label"],cv2.IMREAD_GRAYSCALE)

        img=cv2.resize(img,self.shapeData)
        label=cv2.resize(label,self.shapeData)


        ####Augmentation###
       
        #imgaug=self.TrChannels(img)
        img,label=self.TrFlip(img,label)
        img=self.colorJitter(img)
        img=self.TrBlur(img)
        img=self.TrNoisy(img)
        ###################

        if self.imgTransforms is not None:
            img_o = self.imgTransforms(img)
            imgs = img_o
        else:
            imgs = img

        if self.labelTransforms is not None:
            label_o = self.labelTransforms(label)
            labels = label_o
        else:
            labels = label

        #maskedImg=cv2.copyTo(img,label)
        #cv2.imshow("img",img)
        #cv2.imshow("maskedImg",maskedImg)
        #cv2.imshow("label",label)
        #cv2.waitKey()
        return imgs,labels

    def checkDataset(self):
        for i in range(20):
            self.__getitem__(i)





def checkSegmentationDatasetAnyMaskFilesBinaryClass():

    img_dir="D:\\segmentation\\bacteria\\filtered\\debug"

    input_transform = Compose([
         #transforms.ToPILImage(),
       
         #transforms.Resize((256, 256),Image.BILINEAR),
         transforms.ToTensor(),
         #CreateInputTensorRGBD()
         #Normalize([.485, .456, .406], [.229, .224, .225]),

     ])
    target_transform = Compose([
         #transforms.ToPILImage(),
         #transforms.Resize((256, 256),Image.NEAREST),
         #ToSP(256),
         #ToLabel(),
         #ReLabel(255, 21),
         ])

    dataset=SegmentationDatasetAnyMaskFilesBinaryClass(img_dir,img_transofrm=input_transform,label_transform=target_transform)
    trainloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)

    for _, data in enumerate(trainloader):
         input_net, labels,weight_map = data

         print("input_net: ",input_net.shape)
         print("labels: ",labels.shape)
         print("weight_map: ",weight_map.shape)
         for i in range(1):
            lb=labels[i].numpy()
            img=input_net[i].transpose(2,0).transpose(0,1).numpy()
            wmap=weight_map[i].numpy()

            viewmap=cv2.normalize(wmap,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
            img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC3)

            print("img: ",img.shape)
            print("lb: ",lb.shape)

            cv2.imshow("lb",lb.astype(np.uint8)*100)
            cv2.imshow("img",img.astype(np.uint8))
            cv2.imshow("viewmap",viewmap)

            #iiim=cv2.imread('D:\\segmentation\\bacteria\\filtered\\debug\\0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9\\images\\0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png',cv2.IMREAD_COLOR)
            #cv2.imshow("iiim",iiim)
            cv2.waitKey()

def checkSegmentationDatasetAnyMaskFilesBinaryClassNotMap():

    img_dir="D:\\segmentation\\bacteria\\filtered\\debug"

    input_transform = Compose([
         #transforms.ToPILImage(),
       
         #transforms.Resize((256, 256),Image.BILINEAR),
         transforms.ToTensor(),
         #CreateInputTensorRGBD()
         transforms.Normalize([.485, .456, .406], [.229, .224, .225]),

     ])
    target_transform = Compose([
         #transforms.ToPILImage(),
         #transforms.Resize((256, 256),Image.NEAREST),
         #ToSP(256),
         #ToLabel(),
         #ReLabel(255, 21),
         ])

    dataset=SegmentationDatasetAnyMaskFilesBinaryClassNotMap(img_dir,(256,256),img_transofrm=input_transform,label_transform=target_transform)
    trainloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0)

    for _, data in enumerate(trainloader):
         input_net, labels = data

         print("input_net: ",input_net.shape)
         print("labels: ",labels.shape)

         for i in range(1):
            lb=labels[i].numpy()
            img=input_net[i].transpose(2,0).transpose(0,1).numpy()

            img=cv2.normalize(img,None,0,255,cv2.NORM_MINMAX,cv2.CV_8UC3)

            print("img: ",img.shape)
            print("lb: ",lb.shape)

            cv2.imshow("lb",lb.astype(np.uint8)*100)
            cv2.imshow("img",img.astype(np.uint8))

            #iiim=cv2.imread('D:\\segmentation\\bacteria\\filtered\\debug\\0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9\\images\\0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9.png',cv2.IMREAD_COLOR)
            #cv2.imshow("iiim",iiim)
            cv2.waitKey()

if __name__ == "__main__":
    
    dataset=DatasetInstrumental("D:\\segmentation\\instrumentSegmentation",(320,256))
    dataset.checkDataset()
    #checkSegmentationDatasetAnyMaskFilesBinaryClassNotMap()
    #checkSegmentationDatasetAnyMaskFilesBinaryClass()

