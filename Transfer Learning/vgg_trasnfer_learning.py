import torch
import torch.nn as nn
from torchvision import transforms, models
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset
from glob import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CDDataset(Dataset):
    def __init__(self, folder):
        cats = glob(folder + "\cats\*.jpg")
        dogs = glob(folder + "\dogs\*.jpg")
        self.paths = cats + dogs
        self.normalize = transforms.Normalize(mean = [0.485,0.456,0.406], std = [0.229, 0.224, 0.225])
        self.targets = [float(fpath.split("\\")[-1].startswith('dog')) for fpath in self.paths]
        
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, ix):
        file = self.paths[ix]
        target = self.targets[ix]
        im = (cv2.imread(file)[:,:,::-1])
        im  = cv2.resize(im, (224,224))
        im = torch.tensor(im)/255
        im = im.permute(2,0,1)
        im = self.normalize(im)
        return (im.float().to(device), torch.tensor([target]).to(device))

class VGGnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(weights=models.vgg.VGG16_Weights.DEFAULT)
        for params in self.model.parameters():
            params.requires_grad = False
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512,128), nn.ReLU(), \
                          nn.Dropout(0.2), nn.Linear(128,1), nn.Sigmoid())
    
    def forward(self,x):
        return self.model(x)


training_dir = os.path.join(os.getcwd(), "archive", "training_set", "training_set")
traindata = CDDataset(training_dir)
trainloader = DataLoader(traindata, batch_size = 32, shuffle = True)

test_dir = os.path.join(os.getcwd(), "archive", "test_set", "test_set")
testdata = CDDataset(test_dir)
testloader = DataLoader(testdata, batch_size = 32, shuffle = True)

model = VGGnet().to(device)
#print(summary(model, (3,224,224))) ##checking training parameters
lossfn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
epochs = 5

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(epochs):
    #Train
    model.train()
    train_epoch_losses = []
    train_epoch_accuracies = []
    for x,y in trainloader:
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        correct = torch.sum(torch.round(yhat) == y).item()
        train_epoch_accuracies.append(correct)
        optimizer.zero_grad()
        loss = lossfn(yhat,y)
        loss.backward()
        optimizer.step()
        train_epoch_losses.append(loss.item())
    train_losses.append(np.array(train_epoch_losses).mean())
    train_accuracies.append(np.array(train_epoch_accuracies).sum())
    #Test
    model.eval()
    test_epoch_losses = []
    test_epoch_accuracies = []
    for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            yhat = model(x)
            loss = lossfn(yhat,y)
            test_epoch_losses.append(loss.item())
            correct = torch.sum(torch.round(yhat) == y).item()
            test_epoch_accuracies.append(correct)
    test_accuracies.append(np.array(test_epoch_accuracies).sum())
    test_losses.append(np.array(test_epoch_losses).mean())
    print(f"Epoch({epoch+1}) : Train loss = {train_losses[-1]:.2f} || \
    Train Accuracy = {train_accuracies[-1]/len(traindata):.3f} || \
    Test loss = {test_losses[-1]:.2f} || \
    Test Accuracy = {test_accuracies[-1]/len(testdata):.3f} ")
torch.save(model.state_dict(), "vgg_final_model.pth")

fig, axs = plt.subplots(1, 2)
axs[0].plot(np.arange(epochs)+1, train_losses, label = "Training")
axs[0].plot(np.arange(epochs)+1, test_losses, label = "Validation")
axs[0].set_title('Loss')
axs[0].legend()

axs[1].plot(np.arange(epochs)+1, np.array(train_accuracies)*100/len(traindata), label = "Training")
axs[1].plot(np.arange(epochs)+1, np.array(test_accuracies)*100/len(testdata), label = "Validation")
axs[1].set_title('Accuracy')
axs[1].legend()

plt.tight_layout()
plt.savefig('VGG_Epochs.png')