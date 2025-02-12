from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch

class BTDataset(Dataset):
    def __init__(self,df,img_dir):
        self.h = 224
        self.w = 224
        self.img_dir =img_dir
        self.df = df
        self.image_names = df.ImageID.unique()
        self.label2target = {l:t+1 for t,l in enumerate(self.df['LabelName'].unique())}
        self.label2target['background'] = 0
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self,ix):
        imgid = self.image_names[ix]
        imgpath = os.path.join(self.img_dir, f"{imgid}.jpg")
        img = Image.open(imgpath).convert("RGB")
        img = np.array(img.resize((self.w,self.h),resample = Image.BILINEAR))/255.
        img = torch.tensor(img).permute(2,0,1)
        data = self.df[self.df['ImageID'] == imgid]
        labels = data['LabelName'].values.tolist()
        data = data[['XMin','YMin','XMax','YMax']].values
        data[:,[0,2]]*=self.w
        data[:,[1,3]]*=self.h
        boxes = data.astype(np.uint32).tolist()
        target = {}
        target['boxes'] = torch.tensor(boxes).float()
        target['labels'] = torch.tensor([self.label2target[i] for i in labels]).long()
        return img.float(),target
    
    def collate_fn(self,batch):
        return tuple(zip(*batch))


def train_batch(inputs,model,optimizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    input_, targets = inputs
    input_ = list(image.to(device) for image in input_)
    targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(input_,targets)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss,losses


def val_batch(inputs,model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train() #### "WEIRD" 
    input_, targets = inputs
    input_ = list(image.to(device) for image in input_)
    targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
    optimizer.zero_grad()
    losses = model(input_,targets)
    loss = sum(loss for loss in losses.values())
    return loss,losses