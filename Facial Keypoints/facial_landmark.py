import torch
import os
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import pandas as pd
import cv2
from copy import deepcopy
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FaceDataset(Dataset):
    def __init__(self, root, train= True):
        if train:
            self.folder = os.path.join(root, "training")
            self.csv_file = pd.read_csv(os.path.join(root, "training_frames_keypoints.csv"))
        else:
            self.folder = os.path.join(root, "test")
            self.csv_file = pd.read_csv(os.path.join(root, "test_frames_keypoints.csv"))
        self.normalize = transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, ix):
        img_path = os.path.join(self.folder, self.csv_file.iloc[ix,0])
        img = cv2.imread(img_path)/255
        kp = deepcopy(self.csv_file.iloc[ix, 1:].tolist())
        kp_x = (np.array(kp[0::2]) / img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2]) / img.shape[0]).tolist()
        kp2 = torch.tensor(kp_x + kp_y)
        img = cv2.resize(img, (224,224))
        img = torch.tensor(img).permute(2,0,1)
        img = self.normalize(img).float()
        return img,kp2
    
    def load_img(self,ix):
        img_path = os.path.join(self.folder, self.csv_file.iloc[ix,0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
        img = cv2.resize(img,(224,224))
        return img

class VGGnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(weights=models.vgg.VGG16_Weights.DEFAULT)
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.avgpool = nn.Sequential(nn.Conv2d(512,512,3), nn.MaxPool2d(2),nn.Flatten())
        self.model.classifier = nn.Sequential(nn.Linear(2048,512), nn.ReLU(), nn.Dropout(0.4), nn.Linear(512,136), nn.Sigmoid())
    
    def forward(self,x):
        return self.model(x)

val_imgs = "val_img"
if not os.path.exists(val_imgs):
    os.makedirs(val_imgs)

root_dir = os.path.join(os.getcwd(), "data")
traindata = FaceDataset(root_dir, train=True)
trainloader = DataLoader(traindata, batch_size = 32, shuffle = True)
testdata = FaceDataset(root_dir, train = False)
testloader = DataLoader(testdata, batch_size = 32, shuffle = True)

model = VGGnet().to(device)
#print(summary(model, (3,224,224))) ##checking training parameters
lossfn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
epochs = 50

train_losses = []
test_losses = []
for epoch in range(epochs):
    #Train
    model.train()
    train_epoch_losses = []
    train_epoch_accuracies = []
    for ix, (x,y) in enumerate(trainloader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        loss = lossfn(yhat,y)
        loss.backward()
        optimizer.step()
        train_epoch_losses.append(loss.item())
    train_losses.append(np.array(train_epoch_losses).sum()/(ix+1))
    #Test
    model.eval()
    test_epoch_losses = []
    test_epoch_accuracies = []
    for ix, (x,y) in enumerate(testloader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            yhat = model(x)
            loss = lossfn(yhat,y)
            test_epoch_losses.append(loss.item())
        if(ix == 0):
            plt.figure(figsize = (10,10))
            plt.subplot(221)
            plt.title(f"Original Image")
            img = testdata.load_img(0)
            plt.imshow(img)
            plt.grid(False)
            plt.subplot(222)
            plt.title(f"Image with FL")
            x,_ = testdata[0]
            kp_pred = model(x[None].to(device)).flatten().detach().cpu()
            plt.imshow(img)
            plt.scatter(kp_pred[:68]*224, kp_pred[68:]*224, c = 'r')
            plt.grid(False)
            plt.savefig(os.path.join(val_imgs, f"{epoch+1}.png"))
            plt.close()
    test_losses.append(np.array(test_epoch_losses).sum()/(ix+1))
    print(f"Epoch({epoch+1}) : Train loss = {train_losses[-1]:.4f} || Test loss = {test_losses[-1]:.4f}")
torch.save(model.state_dict(), "vgg_final_model.pth")


plt.plot(np.arange(epochs)+1, train_losses, label = "Training")
plt.plot(np.arange(epochs)+1, test_losses, label = "Validation")
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('FL_Epochs.png')