from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import selectivesearch
import torch
from model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BusTruckDataset(Dataset):
    def __init__(self, df, img_folder):
        self.root = img_folder
        self.df = df
        self.unique_imgs = df['ImageID'].unique()

    def __len__(self):
        return len(self.unique_imgs)
    
    def __getitem__(self, ix):
        img_id = self.unique_imgs[ix]
        img_path = f"{self.root}\\{img_id}.jpg"
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h,w,_ = img.shape
        img_df = self.df[self.df['ImageID'] == img_id]
        boxes = img_df['XMin,YMin,XMax,YMax'.split(',')].values
        boxes = (boxes*np.array([w,h,w,h])).astype(np.uint16).tolist()
        classes = img_df['LabelName'].values.tolist()
        return img, boxes, classes, img_path


class RCNNDataset(Dataset):
    def __init__(self,fpaths,rois,labels, deltas,gtbbs):
        self.fpaths = fpaths
        self.rois = rois
        self.labels = labels
        self.deltas = deltas
        self.gtbbs = gtbbs        
    
    def __len__(self):
        return len(self.fpaths)
    
    def __getitem__(self,ix):
        fpath = str(self.fpaths[ix])
        image = cv2.imread(fpath,1)[...,::-1]
        H,W,_ = image.shape
        sh = np.array([W,H,W,H])
        gtbbs = self.gtbbs[ix]
        rois = self.rois[ix]
        bbs = (np.array(rois)*sh).astype(np.uint16)
        labels = self.labels[ix]
        deltas = self.deltas[ix]
        crops = [image[y:Y,x:X] for x,y,X,Y in bbs]
        return image,crops,bbs,labels,deltas,gtbbs,fpath
    
    def collate_fn(self,batch):
        input_,rois, rixs,labels,deltas = [],[],[],[],[]
        for ix in range(len(batch)):
            image,crops,img_bbs,img_labels,img_deltas,img_gtbbs,fpath = batch[ix]
            crops = [cv2.resize(crop, (224,224)) for crop in crops]
            crops = [preprocess_image(crop/255.)[None] for crop in crops]
            input_.extend(crops)
            labels.extend([label2target[c] for c in img_labels])
            deltas.extend(img_deltas)
        input_ = torch.cat(input_)
        labels = torch.Tensor(labels).long()
        deltas = torch.Tensor(deltas).float()
        return input_, labels, deltas
    
    
def preprocess_image(img):
    img = torch.tensor(img).permute(2,0,1)
    img = normalize(img)
    return img

def decode(_y):
    _, preds = _y.max(-1)
    return preds
    
def extract_candidate(img):
    _,regions = selectivesearch.selective_search(img, scale = 200, min_size = 100)
    img_area = np.prod(img.shape[:2])
    candidates =[]
    for reg in regions:
        if reg['rect'] in candidates:
            continue
        if reg['size'] < (0.05 * img_area):
            continue
        if reg['size'] > img_area:
            continue
        candidates.append(list(reg['rect']))
    return candidates


def cacl_iou(boxA, boxB, epsilon = 1e-5):
    x1 = max(boxA[0],boxA[0])
    y1 = max(boxA[1],boxA[1])
    x2 = min(boxA[2],boxA[2])
    y2 = min(boxA[3],boxA[3])
    width = (x2-x1)
    height= (y2-y1)
    if((width < 0) or (height < 0)):
        return 0.0
    inter = width*height
    areaA= (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB= (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    iou = inter/ (areaA + areaB - inter + epsilon)
    return iou

normalize = transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229,0.224,0.225])
show_sample = False

import pandas as pd
DF_raw = pd.read_csv("archive\\df.csv")
#print(DF_raw.head())
dataset = BusTruckDataset(DF_raw, "archive\\images\\images")
if(show_sample):
    import matplotlib.pyplot as plt
    img_, boxes_, classes_, _ = dataset[9]
    for bbox in boxes_:
        img_ = cv2.rectangle(img_, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
    plt.imshow(img_)
    plt.show()
FPATHS_, GTBBS_, CLSS_, DELTAS_, ROIS_, IOUS_ = [],[],[],[],[],[]
N = 50
for i , (im,bbs,labels,fpath) in enumerate(dataset):
    if(i == N):
        break
    H,W,_ = im.shape
    candidates = extract_candidate(im)
    candidates = np.array([(x,y,x+w,y+h) for x,y,w,h in candidates])
    ious, rois, clss, deltas = [], [], [], []
    ious = np.array([[cacl_iou(candidate, bb) for candidate in candidates] for bb in bbs]).T
    for j, candidate in enumerate(candidates):
        cx,cy,cX,cY = candidate
        candidate_ious = ious[j]
        best_iou_index = np.argmax(candidate_ious)
        best_iou = candidate_ious[best_iou_index]
        bbx, bby, bbX, bbY = bbs[best_iou_index]
        if best_iou > 0.3 :
            clss.append(labels[best_iou_index])
        else:
            clss.append('background')
        delta = np.array([bbx-cx, bby - cy, bbX - cX, bbY - cY])/ np.array([W,H,W,H])
        deltas.append(delta)
        rois.append(candidate/np.array([W,H,W,H]))
    FPATHS_.append(fpath)
    GTBBS_.append(bbs)
    CLSS_.append(clss)
    DELTAS_.append(deltas)
    ROIS_.append(rois)
    IOUS_.append(ious)
#FPATHS_, GTBBS_, CLSS_, DELTAS_, ROIS_, = [item for item in [FPATHS_, GTBBS_, CLSS_, DELTAS_, ROIS_]]
targets = pd.DataFrame([cat for clss in CLSS_ for cat in clss], columns=['label'])
label2target = {l:t for t,l in enumerate(targets['label'].unique())}
target2label = {t:l for l,t in label2target.items()}
backgroundclass = label2target['background']

n_train = 9*len(FPATHS_)//10
train_ds = RCNNDataset(FPATHS_[:n_train], ROIS_[:n_train], CLSS_[:n_train], DELTAS_[:n_train], GTBBS_[:n_train])
test_ds = RCNNDataset(FPATHS_[n_train:], ROIS_[n_train:], CLSS_[n_train:], DELTAS_[n_train:], GTBBS_[n_train:])

train_loader = DataLoader(train_ds, batch_size = 1, collate_fn = train_ds.collate_fn)
test_loader = DataLoader(test_ds, batch_size = 1, collate_fn = train_ds.collate_fn)

rcnn = RCNN().to(device)
creterion = rcnn.cal_loss
optimizer = torch.optim.SGD(rcnn.parameters(), lr = 1e-3)
epochs = 5

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(epochs):
    #Train
    rcnn.train()
    train_epoch_losses = []
    train_epoch_accuracies = []
    for ix,input_ in enumerate(train_loader):
        x,clss,deltas = input_
        optimizer.zero_grad()
        x = x.float().to(device)
        clss = clss.to(device)
        deltas = deltas.to(device)
        _clss,_deltas = rcnn(x)
        loss, locloss, regloss = creterion(_clss,_deltas,clss,deltas)
        accs = clss == decode(_clss)
        loss.backward()
        optimizer.step()
        train_epoch_losses.append(loss.detach().cpu())
        train_epoch_accuracies.append(accs.cpu().numpy().sum())
    train_losses.append(np.array(train_epoch_losses).mean())
    train_accuracies.append(np.array(train_epoch_accuracies).sum())
    
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 1)
axs.plot(np.arange(epochs)+1, train_losses, label = "Training")
plt.show()