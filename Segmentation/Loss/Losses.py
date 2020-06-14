import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss()

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets.long())

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self,predict,target):
        #print(predict)
        smooth=1
        predictView=torch.sigmoid(predict.view(-1))
        targetView=target.view(-1)
        intersection=(predictView*targetView).sum()
        dice_loss=(2.0*intersection+smooth).div(predictView.sum()+targetView.sum()+smooth)
        return 1-dice_loss
 
    
class DiceBCELoss(nn.Module):
    def __init__(self, alpha,betta):
        super(DiceBCELoss, self).__init__()
        self.BCELoss=nn.BCEWithLogitsLoss()
        self.alpha=alpha
        self.betta=betta
    def forward(self, output,target):
        smooth =1.0
        outView=torch.sigmoid(output.view(-1))
        targetView=target.view(-1)
        intersection=(outView*targetView).sum()
        dice=(2.0*intersection+smooth)/(outView.sum()+targetView.sum()+smooth)
        return self.alpha*self.BCELoss(output,target)-self.betta*torch.log(dice)


class JaccardLogitLossBinary(nn.Module):
    def __init__(self,alpha,betta):
        super(JaccardLogitLossBinary, self).__init__()
        self.BCELoss=nn.BCEWithLogitsLoss()
        self.alpha=alpha
        self.betta=betta

    def forward(self, output,target):
        eps=0.000001
        outView=torch.sigmoid(output.view(-1))
        targetView=target.view(-1)
        intersection=(outView*targetView).sum()
        jaccard_value=(intersection+eps)/(outView.sum()+targetView.sum()-intersection+eps)

        loss=self.alpha*self.BCELoss(output,target)-self.betta*torch.log(jaccard_value)
        
        return loss,jaccard_value

class JaccardLossMultiClasses(nn.Module):
    def __init__(self, alpha,betta, num_classes=3,weight=None):
        super(JaccardLossMultiClasses, self,).__init__()
        self.cross_loss = nn.CrossEntropyLoss(weight=weight,ignore_index=255)
        self.alpha=alpha
        self.betta=betta
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        if self.alpha:
            loss = self.alpha*self.cross_loss(outputs, targets)
        else:
            loss=0.0
        eps = 0.000001
        outSoftMax=F.softmax(outputs,dim=1)
        jaccard_val_all_classes=0.0
        for cls in range(self.num_classes):

            jaccard_target = (targets == cls).float()
            
            jaccard_output = outSoftMax[:, cls]

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            
            jaccard_val = (intersection + eps) / (union - intersection + eps)
            jaccard_val_all_classes+=jaccard_val

            loss-=self.betta*torch.log(jaccard_val)

        #print(loss)
        
        return loss,jaccard_val_all_classes/self.num_classes

class JaccardLossMultiClassesFCN101(nn.Module):
    def __init__(self, alpha,betta, num_classes=3):
        super(JaccardLossMultiClassesFCN101, self).__init__()
        self.cross_loss = nn.CrossEntropyLoss()
        self.alpha=alpha
        self.betta=betta
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        if self.alpha:
            loss = self.alpha*self.cross_loss(outputs['out'], targets)
        else:
            loss=0.0
        eps = 0.000001
        outSoftMax=F.softmax(outputs['out'],dim=1)
        jaccard_val_all_classes=0.0
        for cls in range(self.num_classes):

            jaccard_target = (targets == cls).float()
            
            jaccard_output = outSoftMax[:, cls]

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            
            jaccard_val = (intersection + eps) / (union - intersection + eps)
            jaccard_val_all_classes+=jaccard_val

            loss-=self.betta*torch.log(jaccard_val)

        #print(loss)
        
        return loss,jaccard_val_all_classes/self.num_classes

class DiceWithCrossEntropy(nn.Module):
    def __init__(self,num_classes,alpha,beta):
        super(DiceWithCrossEntropy,self).__init__()
        self.crossetropy=nn.CrossEntropyLoss()
        self.alpha=alpha
        self.beta=beta
        self.num_classes=num_classes
    def forward(self,outputs,targets):
        smooth=1.0
        outSoftMax=F.softmax(outputs,dim=1)
        #print(outSoftMax)
        diceLoss=0.0
        for cls in range(self.num_classes):
            cls_target=(targets==cls).float()
            cls_predict=outSoftMax[:,cls]
            intersection=(cls_target*cls_predict).sum()
            diceLoss+=1-(2.0*intersection+smooth).div(cls_target.sum()+cls_predict.sum()+smooth)

        #cla
       
        #generalLoss=diceLoss.div(outputs.shape[1])
        #generalLoss=diceLoss
        #generalLoss=self.crossetropy(outputs,targets.long())
        generalLoss=self.crossetropy(outputs,targets.long())+diceLoss.div(outputs.shape[1])

        return generalLoss

            



if __name__=="__main__":
    input=torch.randn([2,3,3,3],dtype=torch.float)
    target=torch.randint(0,3,(2,3,3))
    print("input: ",input)
    print("target: ",target)
    loss=nn.CrossEntropyLoss(reduction ="sum")
    dice=loss.forward(input,target)
    print(dice)

