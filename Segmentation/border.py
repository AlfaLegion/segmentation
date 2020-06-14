import random
import os
import cv2
import numpy as np
from skimage import measure
from skimage.filters import median
from skimage.morphology import dilation, watershed, square, erosion
from tqdm import tqdm


def create_mask(labels):
        labels = measure.label(labels, neighbors=8, background=0)
        tmp = dilation(labels > 0, square(9))
        tmp2 = watershed(tmp, labels, mask=tmp, watershed_line=True) > 0
        tmp = tmp ^ tmp2
        tmp = dilation(tmp, square(7))
        msk = (255 * tmp).astype('uint8')

        props = measure.regionprops(labels)
        msk0 = 255 * (labels > 0)
        msk0 = msk0.astype('uint8')

        msk1 = np.zeros_like(labels, dtype='bool')

        max_area = np.max([p.area for p in props])

        for y0 in range(labels.shape[0]):
            for x0 in range(labels.shape[1]):
                if not tmp[y0, x0]:
                    continue
                if labels[y0, x0] == 0:
                    if max_area > 4000:
                        sz = 6
                    else:
                        sz = 3
                else:
                    sz = 3
                    if props[labels[y0, x0] - 1].area < 300:
                        sz = 1
                    elif props[labels[y0, x0] - 1].area < 2000:
                        sz = 2
                uniq = np.unique(labels[max(0, y0 - sz):min(labels.shape[0], y0 + sz + 1),
                                 max(0, x0 - sz):min(labels.shape[1], x0 + sz + 1)])
                if len(uniq[uniq > 0]) > 1:
                    msk1[y0, x0] = True
                    msk0[y0, x0] = 0

        msk1 = 255 * msk1
        msk1 = msk1.astype('uint8')

        msk2 = np.zeros_like(labels, dtype='uint8')
        msk = np.stack((msk0, msk1, msk2))
        msk = np.rollaxis(msk, 0, 3)
        return msk1

dataRoot="D:\\datasets\\data-science-bowl-2018\\wich_border\\stage1_train"
listFiles=os.listdir(dataRoot)
dataList=list()
#dataRootBorderMask="D:\\datasets\\data-science-bowl-2018\\stage1_train_border_mask"

for fileName in listFiles:
    fileImages=os.path.join(dataRoot,fileName,"images")
    fileMasks=os.path.join(dataRoot,fileName,"masks")
    fileBorder=os.path.join(dataRoot,fileName,"border")

    listImgs=os.listdir(fileImages)
    listMasks=os.listdir(fileMasks)
    listBorder=os.listdir(fileBorder)

    imgName=os.path.join(fileImages,listImgs[0]),cv2.IMREAD_COLOR
    inputImg=cv2.imread(os.path.join(fileImages,listImgs[0]),cv2.IMREAD_COLOR)
    borderImg=cv2.imread(os.path.join(fileBorder,listBorder[0]),cv2.IMREAD_GRAYSCALE)

    target=np.zeros((inputImg.shape[0],inputImg.shape[1]),dtype=np.uint8)
    
    for mask in listMasks:
        
        maskImg=cv2.imread(os.path.join(fileMasks,mask),cv2.IMREAD_GRAYSCALE)
        target=cv2.bitwise_or(target,maskImg)
        #dataList.append(maskImg)
    
    border_msk=create_mask(target)
    
    #currentPathForSaveBorder=os.path.join(dataRoot,fileName,"border")
    #if not os.path.exists(currentPathForSaveBorder):
    #    os.mkdir(currentPathForSaveBorder)
    #cv2.imwrite(os.path.join(currentPathForSaveBorder,fileName+".png"),border_msk)

    #print(os.path.join(currentPathForSaveBorder,fileName+".png"))

    cv2.imshow("asd",border_msk)
    cv2.imshow("border",border_msk)
    cv2.imshow("inputImg",inputImg)
    cv2.waitKey()