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
import torchvision.models as models

def get_pre_trained_model():
    model=models.segmentation.fcn_resnet101(True)
    
    #print(model)


def validate(model,DataSetLoader):
    torch.cuda.empty_cache()
    model.eval()
    smooth=1.0
    avgDice=[0.0,0.0,0.0]
    num=0
    for input, label in DataSetLoader:
        #input.unsqueeze_(0)
        input=input.cuda()
        label=label.cuda()
        out=model(input)

        outputSoftMax=F.softmax(out,dim=1)
        diceLoss=0.0
        for cls in range(3):
           cls_target=(label==cls).float()
           cls_predict=outputSoftMax[:,cls]
           intersection=(cls_target*cls_predict).sum().item()
           a=cls_target.sum().item()
           b=cls_predict.sum().item()
           diceLoss=(2.0*intersection+smooth)/(cls_target.sum().item()+cls_predict.sum().item()+smooth)
           avgDice[cls]+=diceLoss
        num+=1
    model.train()
    torch.cuda.empty_cache()
    return avgDice[0]/num,avgDice[1]/num,avgDice[2]/num


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

def train(dataRoot,num_epoch,batch_size,lr,momentum):
    
    img_size=(256,256)
    
    transform=augmentation(img_size)
    DataSet=DataLoader.DataLoaderBacteriaBorderWatershed(dataRoot,transform)

    transform_test=albu.Compose([albu.RandomCrop(img_size[0],img_size[1])])
    DataSetTest=DataLoader.DataLoaderBacteriaBorderWatershed(r'D:\datasets\data-science-bowl-2018\wich_border\test',transform_test)


    dataLoader=torch.utils.data.DataLoader(dataset=DataSet,
                                           batch_size=batch_size,
                                           shuffle =True,
                                           num_workers=6)
    dataLoaderTest=torch.utils.data.DataLoader(dataset=DataSetTest,
                                           batch_size=6,
                                           shuffle =False,
                                           num_workers=2)
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
    generalLoss=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="steps",ylable="loss",title="generalLoss"))
    averageLoss=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="steps",ylable="loss",title="averageLoss"))
    DiceTest_c0=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="iteration",ylable="avgDise",title="class 0"))
    DiceTest_c1=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="iteration",ylable="avgDise",title="class 1"))
    DiceTest_c2=vis.line(np.array([0]),np.array([0]),opts=dict(xlabel="iteration",ylable="avgDise",title="class 2"))

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
            
            #draw graphics
            vis.line(np.array([loss.item()]),np.array([general_num]),win=generalLoss,update="append")
            
                     
            general_num+=1
           
            #torch.cuda.empty_cache()

        vis.line(np.array([avg_epoch_loss/num]),np.array([epoch]),win=averageLoss,update="append")
        # test avg dice
        #opt_sheduler.step()
        print("avg loss: ", avg_epoch_loss/num)

        dice_c0,dice_c1,dice_c2=validate(model,dataLoaderTest)

        vis.line(np.array([dice_c0]),np.array([epoch]),win=DiceTest_c0,update="append")  
        vis.line(np.array([dice_c1]),np.array([epoch]),win=DiceTest_c1,update="append")  
        vis.line(np.array([dice_c2]),np.array([epoch]),win=DiceTest_c2,update="append")
        avgDs=(dice_c0+dice_c1+dice_c2)/3
        if avgDs>dice_next and dice_c1>0.85:
        #if(True):
              best_model_wts = copy.deepcopy(model.state_dict())
              dice_next=avgDs

        
    model.load_state_dict(best_model_wts)
    SavePathModel = "./TheBestModels/UNetVGG19BN_2/UNetVGG_"+str(dice_next)+"_.pt"
    convertModel(model,SavePathModel,img_size)


if __name__=="__main__":
    get_pre_trained_model()