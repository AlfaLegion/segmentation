from skimage.segmentation import find_boundaries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import draw
w0 = 10
sigma = 5
import os

def iou(masks_true, masks_pred):
    """
    Get the IOU between each predicted mask and each true mask.

    Parameters
    ----------

    masks_true : array-like
        A 3D array of shape (n_true_masks, image_height, image_width)
    masks_pred : array-like
        A 3D array of shape (n_predicted_masks, image_height, image_width)

    Returns
    -------
    array-like
        A 2D array of shape (n_true_masks, n_predicted_masks), where
        the element at position (i, j) denotes the IoU between the `i`th true
        mask and the `j`th predicted mask.

    """
    if masks_true.shape[1:] != masks_pred.shape[1:]:
        raise ValueError('Predicted masks have wrong shape!')
    n_true_masks, height, width = masks_true.shape
    n_pred_masks = masks_pred.shape[0]
    m_true = masks_true.copy().reshape(n_true_masks, height * width).T
    m_pred = masks_pred.copy().reshape(n_pred_masks, height * width)
    numerator = np.dot(m_pred, m_true)
    denominator = m_pred.sum(1).reshape(-1, 1) + m_true.sum(0).reshape(1, -1)
    return numerator / (denominator - numerator)

def make_weight_map(masks):
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


dataRoot="D:\\datasets\\data-science-bowl-2018\\distance_error\\stage1_train"
listFiles=os.listdir(dataRoot)


for fileName in listFiles:
    fileImages=os.path.join(dataRoot,fileName,"images")
    fileMasks=os.path.join(dataRoot,fileName,"masks")

    listImgs=os.listdir(fileImages)
    listMasks=os.listdir(fileMasks)

    imgName=os.path.join(fileImages,listImgs[0]),cv2.IMREAD_COLOR
    inputImg=cv2.imread(os.path.join(fileImages,listImgs[0]),cv2.IMREAD_COLOR)

    target=np.zeros((inputImg.shape[0],inputImg.shape[1]),dtype=np.uint8)
    maskslist=list()
    for mask in listMasks:
        
        maskImg=cv2.imread(os.path.join(fileMasks,mask),cv2.IMREAD_GRAYSCALE)
        maskslist.append(maskImg)
    

    masks=np.array(maskslist)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
    ax1.imshow(masks.sum(0))
    ax1.set_axis_off()
    ax1.set_title('True Masks', fontsize=15)

    weights = make_weight_map(masks).astype(np.float32)
    print(weights.shape,"   ",masks[0].shape)

    pos = ax2.imshow(weights)
    pos = ax2.imshow(weights)
    ax2.set_axis_off()
    ax2.set_title('Weights', fontsize=15)
    _ = fig.colorbar(pos, ax=ax2)
    plt.show()
    
    #currentPathForSaveBorder=os.path.join(dataRoot,fileName,"border")
    #if not os.path.exists(currentPathForSaveBorder):
    #    os.mkdir(currentPathForSaveBorder)
    #cv2.imwrite(os.path.join(currentPathForSaveBorder,fileName+".png"),border_msk)

    #print(os.path.join(currentPathForSaveBorder,fileName+".png"))

    #cv2.imshow("asd",border_msk)
    #cv2.imshow("border",border_msk)
    #cv2.imshow("inputImg",inputImg)
    #cv2.waitKey()


#11111111111111111111111111

#pathdata="D:\\datasets\\data-science-bowl-2018\\distance_error\\stage1_train\\00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e\\masks"
#listMask=os.listdir(pathdata)
#maskslist=[]
#for nameImg in listMask:
#    mask=cv2.imread(os.path.join(pathdata,nameImg),cv2.IMREAD_GRAYSCALE)
#    mask[mask>0]=1
#    maskslist.append(mask)

#masks=np.array(maskslist)

#fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
#ax1.imshow(masks.sum(0))
#ax1.set_axis_off()
#ax1.set_title('True Masks', fontsize=15)



#import time
#start=time.clock()
#weights = make_weight_map(masks).astype(np.float32)
#end=time.clock()
#print("time: ",end-start)
#pos = ax2.imshow(weights)
#ax2.set_axis_off()
#ax2.set_title('Weights', fontsize=15)
#_ = fig.colorbar(pos, ax=ax2)
#plt.show()
#img=cv2.imread("temp.png",cv2.IMREAD_GRAYSCALE)

#cv2.imshow("azsd",img)
#cv2.waitKey()

