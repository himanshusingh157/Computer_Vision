import torch.nn as nn
from torchvision import models
import torch 

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_backbone = models.vgg16(weights=models.vgg.VGG16_Weights.DEFAULT)
        vgg_backbone.classifier = nn.Sequential()
        for param in vgg_backbone.parameters():
            param.require_grad = False
        self.backbone = vgg_backbone
        feature_dim = 25088
        self.cls_score = nn.Linear(feature_dim,3)
        self.bbox = nn.Sequential(nn.Linear(feature_dim,512), nn.ReLU(), nn.Linear(512,4), nn.Tanh())
        self.CELoss = nn.CrossEntropyLoss()
        self.L1Loss = nn.L1Loss()
    
    def forward(self,x):
        feat = self.backbone(x)
        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat)
        return cls_score, bbox
        
    def cal_loss(self, probs, _deltas, labels, deltas):
        dloss = self.CELoss(probs,labels)
        ixs, = torch.where(labels!=0)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        lmb = 10.0
        if len(ixs) > 0:
            regloss = self.L1Loss(_deltas,deltas)
            return dloss + lmb*regloss, dloss.detach(), regloss.detach()
        else:
            regloss = 0
            return dloss + lmb*regloss, dloss.detach(), regloss

