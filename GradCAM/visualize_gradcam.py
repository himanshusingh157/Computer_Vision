import torch
import os
import random
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import cv2
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn.functional as F
import time
from model import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
valimgs = "val_imgs"
if not os.path.exists(valimgs):
    os.makedirs(valimgs)

def im2gradCAM(model, x, ix):
    logits = model(x)
    pred = logits.max(-1)[-1]
    if(pred.item()):
        activation = model.im2fmap(x)
        model.zero_grad()
        logits[0,pred].backward(retain_graph = True)
        pooled_grad = model.convblock6.conv.weight.grad.data.mean((1,2,3))
        activation *= pooled_grad[:,None,None]
        heatmap = torch.mean(activation,dim = 1)[0].cpu().detach()
        mini, maxi = heatmap.min(), heatmap.max()
        heatmap = (heatmap - mini)/(maxi - mini)
        heatmap = F.interpolate(heatmap[None,None, ...], size=(128, 128), mode='bilinear')
        jetmap = plt.get_cmap('jet')
        heatmap = jetmap(torch.squeeze(heatmap))[:,:,:3]
        heatmap = torch.tensor(heatmap).permute(2, 0, 1).to(device)
        overlayed_img = 0.7*heatmap + 0.3*x
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(x[0].cpu().permute(1, 2, 0))
        ax[1].imshow(overlayed_img[0].cpu().permute(1, 2, 0))
        fig.suptitle("Parasitized" if pred.item() else "Unifected")
        plt.savefig(f"{valimgs}\\{ix}.png")
        plt.close()
    
    

root_folder = os.path.join(os.getcwd(), "archive", "cell_images")
all_files = glob(f"{root_folder}\\*\\*.png")
random.shuffle(all_files)
all_files = all_files[:20]

data = MalariaDataset(all_files, train=False)
loader = DataLoader(data, batch_size = 1, shuffle = True)

model = MalariaClassifier().to(device)
model.load_state_dict(torch.load("final_model.pth", weights_only=True))
model.eval()

for i,(x,y) in enumerate(loader):
    x = x.to(device)
    im2gradCAM(model,x, i)