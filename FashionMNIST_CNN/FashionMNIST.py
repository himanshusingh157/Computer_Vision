#Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import datasets
import matplotlib.pyplot as plt


#Download Dataset if needed
data_folder = "~\data\FMNIST"
fmnist = datasets.FashionMNIST(data_folder, download = True, train = True)
images = fmnist.data
targets = fmnist.targets


#Plot Random images for all classe
R = len(targets.unique())
C = 10
fig, ax = plt.subplots(R,C, figsize = (10,10))
for class_, plt_row in enumerate(ax):
    label_x = np.where(targets == class_)[0]
    for plt_square in plt_row:
        plt_square.grid(False)
        plt_square.axis('off')
        index = np.random.choice(label_x)
        img = images[index]
        plt_square.imshow(img, cmap = 'gray')
plt.tight_layout()
plt.savefig('Data.png')
    
#Defining Datatset    
class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        x = x.float()/255
        self.X = x.view(-1,1,28,28)
        self.Y = y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,index):
        return self.X[index], self.Y[index]

#Defining Model
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16,32,3)
        self.linear1 = nn.Linear(800,128)
        self.linear2 = nn.Linear(128,10)
        self.maxpool = nn.MaxPool2d(2,2)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        y = self.maxpool(self.conv1(x))
        y = torch.relu(y)
        y = self.maxpool(self.conv2(y))
        y = torch.relu(y)
        y = self.flatten(y)
        y = torch.sigmoid(self.linear1(y))
        y = self.softmax(self.linear2(y))
        return y
        
        

#Check Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device Available : {device}")

#Model, Data, Loss, Optimizer
traindata = MyDataset(images[:50000],targets[:50000])
trainloader = DataLoader(traindata, batch_size = 128, shuffle = True)
testdata = MyDataset(images[50000:],targets[50000:])
testloader = DataLoader(testdata, batch_size = 128, shuffle = True)
net = MyNet()
net = net.to(device)

lossfn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr = 0.001)
epochs = 100

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(epochs):
    #Train
    net.train()
    train_epoch_losses = []
    train_epoch_accuracies = []
    for x,y in trainloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        _, argmaxes = yhat.max(-1)
        correct = argmaxes == y
        train_epoch_accuracies.append(correct.sum().item())
        optim.zero_grad()
        loss = lossfn(yhat,y)
        loss.backward()
        optim.step()
        train_epoch_losses.append(loss.item())
    train_losses.append(np.array(train_epoch_losses).mean())
    train_accuracies.append(np.array(train_epoch_accuracies).sum())
    #Test
    net.eval()
    test_epoch_losses = []
    test_epoch_accuracies = []
    for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        loss = lossfn(yhat,y)
        test_epoch_losses.append(loss.item())
        _, argmaxes = yhat.max(-1)
        correct = argmaxes == y
        test_epoch_accuracies.append(correct.sum().item())
    test_accuracies.append(np.array(test_epoch_accuracies).sum())
    test_losses.append(np.array(test_epoch_losses).mean())
    print(f"Epoch({epoch}) : Train loss = {train_losses[-1]:.2f} || \
    Train Accuracy = {train_accuracies[-1]/50000:.2f} || \
    Test loss = {test_losses[-1]:.2f} || \
    Test Accuracy = {test_accuracies[-1]/10000:.2f} ")
torch.save(net.state_dict(), "final_model.pth")


fig, axs = plt.subplots(1, 2)
axs[0].plot(np.arange(epochs)+1, train_losses, label = "Training")
axs[0].plot(np.arange(epochs)+1, test_losses, label = "Validation")
axs[0].set_title('Loss')
axs[0].legend()

axs[1].plot(np.arange(epochs)+1, np.array(train_accuracies)/500, label = "Training")
axs[1].plot(np.arange(epochs)+1, np.array(test_accuracies)/100, label = "Validation")
axs[1].set_title('Accuracy')
axs[1].legend()

plt.tight_layout()
plt.savefig('Epochs.png')
