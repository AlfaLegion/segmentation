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
            IoUs.append(iou_intersection[cls]/(max(iou_union[cls],1)))
    
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



def trainUNetVGG19BN(train_data,test_train,save_model_name,num_epoch,batch_size,lr,momentum,treshold_f1_class_object=0.85):
    
    img_size=(256,256)
    num_workers_train=6
    num_workers_test=6
    transform=augmentation(img_size)
    DataSet=DataLoader.DataLoaderBacteriaBorderWatershed(train_data,transform)

    transform_test=albu.Compose([albu.RandomCrop(img_size[0],img_size[1])])
    DataSetTest=DataLoader.DataLoaderBacteriaBorderWatershed(test_train,transform_test)


    dataLoader=torch.utils.data.DataLoader(dataset=DataSet,
                                           batch_size=batch_size,
                                           shuffle =True,drop_last =True,
                                           num_workers=num_workers_train)
    dataLoaderTest=torch.utils.data.DataLoader(dataset=DataSetTest,
                                           batch_size=20,
                                           shuffle =False,
                                           num_workers=num_workers_test)
    print("\nInitialization net...")
    model=UNetTransferLerning.UNetVGG(3,3,64).cuda()
    print("Complete")

    loss_function=Losses.DiceWithCrossEntropy(3,1,1)

    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=0.005)
    #optimizer=torch.optim.Adadelta(model.parameters(),lr=lr,rho=0.9,weight_decay=0.005)
    #opt_sheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.9)

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
    for epoch in range(num_epoch):
        print("Epoch: ",epoch+1,"/",num_epoch)
        avg_epoch_loss=0.0
        num=0
        for input,label in dataLoader:
            ########################################
            #import numpy as np
            #a=input[0].numpy().copy()
            #a=a.transpose((2,1,0)).copy()
            
            #a=a.transpose((1,0,2))
            #a=cv2.normalize(a,None,0.,255.,cv2.NORM_MINMAX)
            #cv2.imshow("image",a.astype(dtype=np.uint8))

            #b=label[0].numpy().copy()
            #print(np.unique(b))
            ##b=b.transpose((2,1,0)).copy()
            ##b=b.transpose((1,0,2),)
            #b=cv2.normalize(b,None,0,255,cv2.NORM_MINMAX)

            #cv2.imshow("label",b.astype(dtype=np.uint8))
            #cv2.waitKey()
            ########################################
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
           
            #torch.cuda.empty_cache()

        vis.line(np.array([avg_epoch_loss/num]),np.array([epoch]),win=averageLoss,update="append")
        # test avg dice
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
        if avgDs>dice_next and F1_[1]>treshold_f1_class_object:
        #if(True):
              best_model_wts = copy.deepcopy(model.state_dict())
              dice_next=avgDs

        
    model.load_state_dict(best_model_wts)
    convertModel(model,save_model_name,img_size)

def trainUNetResNet(train_data,test_train,type_resnet,save_model_name,num_epoch,batch_size,lr,momentum,treshold_f1_class_object=0.85):
    
    img_size=(256,256)
    num_workers_train=6
    num_workers_test=6
    transform=augmentation(img_size)
    DataSet=DataLoader.DataLoaderBacteriaBorderWatershed(train_data,transform)

    transform_test=albu.Compose([albu.RandomCrop(img_size[0],img_size[1])])
    DataSetTest=DataLoader.DataLoaderBacteriaBorderWatershed(test_train,transform_test)


    dataLoader=torch.utils.data.DataLoader(dataset=DataSet,
                                           batch_size=batch_size,
                                           shuffle =True,drop_last =True,
                                           num_workers=num_workers_train)
    dataLoaderTest=torch.utils.data.DataLoader(dataset=DataSetTest,
                                           batch_size=6,
                                           shuffle =False,
                                           num_workers=num_workers_test)
    print("\nInitialization net...")
    model=UNetTransferLerning.UNetResNet(3,3,64,type_resnet).cuda()
    print("Complete")

    loss_function=Losses.DiceWithCrossEntropy(3,1,1)

    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=0.005)
    #optimizer=torch.optim.Adadelta(model.parameters(),lr=lr,rho=0.9,weight_decay=0.005)
    #opt_sheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.9)

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
    for epoch in range(num_epoch):
        print("Epoch: ",epoch+1,"/",num_epoch)
        avg_epoch_loss=0.0
        num=0
        for input,label in dataLoader:
            ########################################
            #import numpy as np
            #a=input[0].numpy().copy()
            #a=a.transpose((2,1,0)).copy()
            
            #a=a.transpose((1,0,2))
            #a=cv2.normalize(a,None,0.,255.,cv2.NORM_MINMAX)
            #cv2.imshow("image",a.astype(dtype=np.uint8))

            #b=label[0].numpy().copy()
            #print(np.unique(b))
            ##b=b.transpose((2,1,0)).copy()
            ##b=b.transpose((1,0,2),)
            #b=cv2.normalize(b,None,0,255,cv2.NORM_MINMAX)

            #cv2.imshow("label",b.astype(dtype=np.uint8))
            #cv2.waitKey()
            ########################################
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
           
            #torch.cuda.empty_cache()

        vis.line(np.array([avg_epoch_loss/num]),np.array([epoch]),win=averageLoss,update="append")
        # test avg dice
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

        if avgDs>dice_next and F1_[1]>treshold_f1_class_object:
        #if(True):
              best_model_wts = copy.deepcopy(model.state_dict())
              dice_next=avgDs

        
    model.load_state_dict(best_model_wts)
    convertModel(model,save_model_name,img_size)


def visdom_exmpl():
    
    vis = visdom.Visdom()
    plot=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="x",ylable="y"))
    for x in range(1,100):
        xx=np.array([x])
        yy=np.array([x*x])
        vis.line(xx,yy,win=plot,update="append")

if __name__=="__main__":

    #trainUNetVGG19BN(r'D:\datasets\data-science-bowl-2018\wich_border_2\stage1_train',
    #r'D:\datasets\data-science-bowl-2018\wich_border_2\test',
    #'./TheBestModels/d2/UNetVGG19BN/UNetVGG.pt',
    #      num_epoch=1,batch_size=9,lr=0.0004,momentum=0.95,treshold_f1_class_object=0.01)

    #trainUNetResNet(r'D:\datasets\data-science-bowl-2018\wich_border_2\stage1_train',
    #                r'D:\datasets\data-science-bowl-2018\wich_border_2\test',
    #                'resnet34',
    #                "./TheBestModels/d2/UNetResNet34/UNetResNet34.pt",
    #      num_epoch=1,batch_size=6,lr=0.0004,momentum=0.95,treshold_f1_class_object=0.01)




