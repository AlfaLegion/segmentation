import torch
import albumentations as albu
from Loss import Losses
from UNet import Unet  
from UNet import UNetTransferLerning  
from DataLoader import DataLoaderBacteriaBorderWatershed as DataLoader
from torch.autograd import Variable
import cv2
from ConvertModels.convert import*
import visdom
import numpy as np
import torch.nn.functional as F
import copy
import time as t


def F1_IoU(model,DataSetLoader):

    model.eval()
    smooth=1.0
    f1_intersection=[0.0,0.0,0.0]
    f1_total=[0.0,0.0,0.0]
    F1_score=list()

    iou_intersection=[0.0,0.0,0.0]
    iou_union=[0.0,0.0,0.0]
    IoUs=list()
    with torch.no_grad():
        for input, label in DataSetLoader:
            label=label.cuda()
            input=input.cuda()
            outputSoftMax=F.softmax(model(input),dim=1)
            argmaxPredict=torch.argmax(outputSoftMax,dim=1)
            for cls in range(3):              
                cls_target=(label==cls)

                # for f1 score
                cls_target_f1=cls_target.float()
                cls_predict_f1=outputSoftMax[:,cls]                   
                f1_intersection[cls]+=(cls_target_f1*cls_predict_f1).sum().item()
                f1_total[cls]+=(cls_target_f1.sum().item()+cls_predict_f1.sum().item()+smooth)

                # for IoU
                cls_predict_iou=(argmaxPredict==cls).long()

                cls_target_iou=cls_target.view(-1).long()
                cls_predict_iou=cls_predict_iou.view(-1)

                iou_intersection[cls]+=float((cls_target_iou & cls_predict_iou).sum().item())
                iou_union[cls]+=float((cls_target_iou | cls_predict_iou).sum().item())


        for cls in range(3):
            F1_score.append((2*f1_intersection[cls]+smooth)/(f1_total[cls]+smooth))
            IoUs.append(iou_intersection[cls]/max(iou_union[cls],1.0))
    
    model.train()
    torch.cuda.empty_cache()
    return F1_score,IoUs


def augmentation(img_size):
    return albu.Compose([
        #spatial wise
        #albu.RandomCrop(img_size[0],img_size[1]),
        albu.RandomResizedCrop(img_size[0],img_size[1],
                               interpolation=cv2.INTER_NEAREST),
        albu.Rotate(interpolation=cv2.INTER_NEAREST),
        albu.HorizontalFlip(),
        albu.ElasticTransform(alpha_affine=10,interpolation=cv2.INTER_NEAREST),
        albu.OpticalDistortion(interpolation=cv2.INTER_NEAREST),

        #pixel transform
        albu.RandomGamma((100,180),p=0.8),
        albu.Blur(5),
        albu.RandomBrightness(limit=(0.05,0.20)),
        albu.RandomContrast(limit=(0,0.20),p=0.5),
        albu.MotionBlur( blur_limit =7),

        ])


#
#factor_epoch - число эпох, когда нужно увеличить размер партии на betta_inc_bs
#max_batch_size - предельный размер партии.
#
def trainUNet(train_data,test_data,type_coder,save_model_name,num_epoch,start_batch_size,lr,momentum,betta_inc_bs,gamma_decay_lr,treshold_f1_class_object=0.85,factor_epoch=10,max_batch_size=9):
    
    img_size=(256,256)
    num_workers_train=6
    num_workers_test=6
    factor_epoch=10
    max_batch_size=9

    transform=augmentation(img_size)
    DataSet=DataLoader.DataLoaderBacteriaBorderWatershed(train_data,transform)

    transform_test=albu.Compose([albu.RandomCrop(img_size[0],img_size[1])])
    DataSetTest=DataLoader.DataLoaderBacteriaBorderWatershed(test_data,transform_test)


    dataLoader=torch.utils.data.DataLoader(dataset=DataSet,
                                           batch_size=start_batch_size,
                                           shuffle =True,
                                           num_workers=num_workers_train)
    dataLoaderTest=torch.utils.data.DataLoader(dataset=DataSetTest,
                                           batch_size=20,
                                           shuffle =False,
                                           num_workers=num_workers_test)
    print("\nInitialization net...")

    if type_coder=='vgg19bn':
        model=UNetTransferLerning.UNetVGG(3,3,64).cuda()
    else:
        model=UNetTransferLerning.UNetResNet(3,3,64,type_coder).cuda()
    
    
    print("Complete")

    loss_function=Losses.DiceWithCrossEntropy(3,1,1)

    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=0.0005)
   
    if gamma_decay_lr is not None:
        opt_sheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=gamma_decay_lr)
    new_batch_size=start_batch_size
    model.train()
    print("\nTrain...\n")

    vis = visdom.Visdom()
    averageLoss=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="loss",title="Average Loss"))

    F1Test_c0=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="F1",title="F1 background"))
    F1Test_c1=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="F1",title="F1 objects"))
    F1Test_c2=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="F1",title="F1 line watershed"))

    IoUTest_c0=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="IoU",title="IoU background"))
    IoUTest_c1=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="IoU",title="IoU objects"))
    IoUTest_c2=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="IoU",title="IoU line watershed"))

    MeanIoUGraph=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="epoch",ylable="MeanIoU",title="MeanIoU"))
    general_num=0
    dice_next=0.0
    all_train_time=0.0
    best_iou=0.0
    for epoch in range(num_epoch):
        print("Epoch: ",epoch+1,"/",num_epoch)
        avg_epoch_loss=0.0
        num=0

        #update bacth size per 30 epoch
       
        
        start_time=t.time()
        for input,label in dataLoader:
           
            optimizer.zero_grad()


            input=Variable(input).cuda()
            label=Variable(label).cuda()

            output=model(input)
            loss=loss_function(output,label)
            
            #print("current loss: ",loss.item())   
            avg_epoch_loss+=loss.item()
            num+=1
            loss.backward()
            optimizer.step()
                  
            general_num+=1
           
            
        end_time=t.time()
        all_train_time+=(end_time-start_time)
        

        vis.line(np.array([avg_epoch_loss/num]),np.array([epoch]),win=averageLoss,update="append")
        

        #update batch size

        if (epoch+1)%factor_epoch==0 and epoch!=0:
           new_batch_size=betta_inc_bs+new_batch_size
           if new_batch_size>max_batch_size:
               new_batch_size=max_batch_size
           dataLoader=torch.utils.data.DataLoader(dataset=DataSet,
                                          batch_size=round(new_batch_size),
                                          shuffle =True,
                                          drop_last=True,
                                          num_workers=num_workers_train)
        #update larning rate
        #opt_sheduler.step()

        print("avg loss: ", avg_epoch_loss/num)

        F1_,IoU_=F1_IoU(model,dataLoaderTest)
        MeanIoUValue=(IoU_[0]+IoU_[1]+IoU_[2])/3.0
        print("f1: ",F1_)
        print("iou: ",IoU_)
        print("MeanIoU: ",MeanIoUValue,'\n')
      
        #draw graphics
        vis.line(np.array([F1_[0]]),np.array([epoch]),win=F1Test_c0,update="append")  
        vis.line(np.array([F1_[1]]),np.array([epoch]),win=F1Test_c1,update="append")  
        vis.line(np.array([F1_[2]]),np.array([epoch]),win=F1Test_c2,update="append")

        vis.line(np.array([IoU_[0]]),np.array([epoch]),win=IoUTest_c0,update="append")  
        vis.line(np.array([IoU_[1]]),np.array([epoch]),win=IoUTest_c1,update="append")  
        vis.line(np.array([IoU_[2]]),np.array([epoch]),win=IoUTest_c2,update="append")

        vis.line(np.array([MeanIoUValue]),np.array([epoch]),win=MeanIoUGraph,update="append")


        avgDs=(F1_[0]+F1_[1]+F1_[2])/3
        if best_iou>F1_[1]:
            best_iou=F1_[1]
        if avgDs>dice_next and F1_[1]>treshold_f1_class_object:
        #if(True):
              best_model_wts = copy.deepcopy(model.state_dict())
              dice_next=avgDs

    print("Full time of train: ",all_train_time," sec")
    model.load_state_dict(best_model_wts)
    convertModel(model,save_model_name,img_size)


if __name__=="__main__":
   
    trainUNet(r'D:\datasets\data-science-bowl-2018\wich_border_2\stage1_train',
          r'D:\datasets\data-science-bowl-2018\wich_border_2\test','resnet34',
          "D:/Projects/Segmentation models/TheBestModels/d2/adabatchvgg19/AB_unet_vgg_3.pt",
          num_epoch=300,start_batch_size=1,lr=0.0004,momentum=0.99,betta_inc_bs=1,gamma_decay_lr=None,treshold_f1_class_object=0.82)
   # visdom_exmpl()



