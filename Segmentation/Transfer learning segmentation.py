import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from torchvision import models
import os
import time as t
import Losses
import DataLoader
from torch.autograd import Variable

def get_info(model):
    from torchsummary import summary
    print(model)

def convertModel(SavePathModel,newNameModel):
    print("Start...")
    device = torch.device('cpu')
    model = torch.load(SavePathModel,map_location=device)
    model = model.float()
    model.eval()
    example = torch.rand(1, 3, 256, 256)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save(newNameModel)
    print("Complete")


def train():
    model = models.segmentation.fcn_resnet101(pretrained=True).float()
    model.classifier[4]=nn.Conv2d(512,3,(1,1),(1,1))
    model.aux_classifier[4]=nn.Conv2d(256,3,kernel_size=(1, 1),stride=(1, 1))

    batch_size = 10
    img_size = 256
    lr = 0.008
    epoch = 15
    
    img_dir = "D:\\segmentation\\classifySegmentationTwoClasses256\\"

    input_transform = transforms.Compose([#transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])])
    target_transform = transforms.Compose([DataLoader.ToLabelMultyClasses()])

    img_data = DataLoader.SegmentationDataset(img_dir,img_transofrm=input_transform,label_transform=target_transform)
    img_batch = torch.utils.data.DataLoader(img_data,batch_size=batch_size,shuffle=True,num_workers=0)

    loss_func = Losses.JaccardLossMultiClassesFCN101(1,0.9)
    gen_optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.85,weight_decay=0.0005)

    model.train()
    model.cuda(0)
    print("\nTrain...\n")
    for i in range(epoch):
        print("Epoch: ",i + 1,"/",epoch)
        avg_loss = 0.0
        avg_IoU = 0.0

        for num,(images,labels) in enumerate(img_batch):

            gen_optimizer.zero_grad()
            
            input = Variable(images).cuda(0)
            labels = Variable(labels).cuda(0)
    
            output = model(input) 

            loss,jaccard = loss_func(output,labels)

            avg_IoU+=jaccard.item()
            avg_loss+=loss.item()

            loss.backward()
            gen_optimizer.step()

        print("number of iterations per epoch: ",num)    
        print("avg loss: ",avg_loss / num)
        print("avg IoU: ",avg_IoU / num,"\n")

    SavePathModel = "SegModelFCNResNet101_transfer.pt"
    
    torch.save(model,SavePathModel)
    convertModel(SavePathModel,"SegModelFCNResNet101_transfer_script.pt")


    

train()
#model = models.segmentation.fcn_resnet101(pretrained=True).float()
#get_info(model)
