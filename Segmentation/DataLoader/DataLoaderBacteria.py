import torch
import cv2
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import TranformsData.Transforms as trans
import albumentations as aldu


def make_weight_map(masks,w0,sigma):
    """
    Generate the weight maps as specified in the UNet paper
    for a set of binary masks.
    
    Parameters
    ----------
    masks: array-like
        A 3D array of shape (n_masks, image_height, image_width),
        where each slice of the matrix along the 0th axis represents one binary mask.

    Returns
    -------
    array-like
        A 2D array of shape (image_height, image_width)
    
    """
    nrows, ncols = masks.shape[1:]
    masks = (masks > 0).astype(int)
    distMap = np.zeros((nrows * ncols, masks.shape[0]))
    X1, Y1 = np.meshgrid(np.arange(nrows), np.arange(ncols))
    X1, Y1 = np.c_[X1.ravel(), Y1.ravel()].T
    for i, mask in enumerate(masks):
        # find the boundary of each mask,
        # compute the distance of each pixel from this boundary
        bounds = find_boundaries(mask, mode='inner')
        X2, Y2 = np.nonzero(bounds)
        xSum = (X2.reshape(-1, 1) - X1.reshape(1, -1)) ** 2
        ySum = (Y2.reshape(-1, 1) - Y1.reshape(1, -1)) ** 2
        distMap[:, i] = np.sqrt(xSum + ySum).min(axis=0)
    ix = np.arange(distMap.shape[0])
    if distMap.shape[1] == 1:
        d1 = distMap.ravel()
        border_loss_map = w0 * np.exp((-1 * (d1) ** 2) / (2 * (sigma ** 2)))
    else:
        if distMap.shape[1] == 2:
            d1_ix, d2_ix = np.argpartition(distMap, 1, axis=1)[:, :2].T
        else:
            d1_ix, d2_ix = np.argpartition(distMap, 2, axis=1)[:, :2].T
        d1 = distMap[ix, d1_ix]
        d2 = distMap[ix, d2_ix]
        border_loss_map = w0 * np.exp((-1 * (d1 + d2) ** 2) / (2 * (sigma ** 2)))
    xBLoss = np.zeros((nrows, ncols))
    xBLoss[X1, Y1] = border_loss_map
    # class weight map
    loss = np.zeros((nrows, ncols))
    w_1 = 1 - masks.sum() / loss.size
    w_0 = 1 - w_1
    loss[masks.sum(0) == 1] = w_1
    loss[masks.sum(0) == 0] = w_0
    ZZ = xBLoss + loss
    return ZZ


class DataLoaderDifferentMasks(torch.utils.data.Dataset):
    def __init__(self,dataRoot,img_prefix,mask_prefix,transform=None):
        listFiles=os.listdir(dataRoot)
        self.dataList=list()
        #Augmentation Object
        self.transform=transform
        #
        self.ToTensor=transforms.ToTensor()
        self.Normalize=transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        self.LabelTotensor=trans.ToLabel()

        for fileName in listFiles:
            fileImages=os.path.join(dataRoot,fileName,img_prefix)
            fileMasks=os.path.join(dataRoot,fileName,mask_prefix)

            listImgs=os.listdir(fileImages)
            listMasks=os.listdir(fileMasks)


            lsMask=list()
            for mask in listMasks:
                lsMask.append(os.path.join(fileMasks,mask))

            self.dataList.append({
                "input":os.path.join(fileImages,listImgs[0]),
                "labels":lsMask})

        print("Number of labels: ",len(self.dataList))

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self,index):
        dataFile=self.dataList[index]

        #read input
        input=cv2.imread(dataFile["input"],cv2.IMREAD_COLOR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        #read targes
        label=np.zeros((input.shape[0],input.shape[1]),dtype=np.uint8)
        for targetPath in dataFile["labels"]:
            LoadMask=cv2.imread(targetPath,cv2.IMREAD_GRAYSCALE)
            label=cv2.bitwise_or(label,LoadMask)
    
        # Augmentation block

        if self.transform:
            augmented=self.transform(image=input,mask=label)
        #input=np.array(input)

        #

        input_tensor=self.ToTensor(augmented["image"])
        input_tensor=self.Normalize(input_tensor)
        label_tensor=self.LabelTotensor(augmented["mask"])

        #return augmented["image"], augmented["mask"]
        return input_tensor,label_tensor

    def checkdataLoader(self):
        print("dummy")


if __name__=="__main__":

    transform=aldu.Compose([
        ###pixel_lvl_transform
        
        aldu.CLAHE(p=1),
        #aldu.RandomBrightnessContrast(),
        aldu.ChannelDropout(),
        #aldu.ISONoise() ,
        #aldu.Downscale(),
        #aldu.MultiplicativeNoise(),
        
        ###spatial lvl transform
        aldu.RandomCrop(256,256),
        aldu.HorizontalFlip(),
        aldu.Rotate(),
        aldu.MaskDropout(),
        aldu.ElasticTransform(),
        aldu.GridDistortion(),
  
        ### normalize
        #aldu.Normalize()
        ])
    dataPath="D:\\segmentation\\bacteria\\stage1_train"
    Dataset=DataLoaderDifferentMasks(dataPath,"images","masks",transform)

    #DataLoader=torch.utils.data.DataLoader(dataset=Dataset,shuffle =True)



    for i in range(100):
       input,target = Dataset[i]

       cv2.imshow("input",input)
       cv2.imshow("target",target)
       cv2.waitKey()