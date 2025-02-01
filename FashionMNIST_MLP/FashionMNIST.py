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
        self.X = x.view(-1,28*28)
        self.Y = y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,index):
        return self.X[index], self.Y[index]

#Defining Model
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784,200)
        self.layer2 = nn.Linear(200,10)
        self.actiavtionfn = nn.Sigmoid()
        
    def forward(self,x):
        y = self.actiavtionfn(self.layer1(x))
        y = self.layer2(y)
        return y

#Check Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"Device Available : {device}")

#Model, Data, Loss, Optimizer
traindata = MyDataset(images[:50000],targets[:50000])
trainloader = DataLoader(traindata, batch_size = 128, shuffle = True)
testdata = MyDataset(images[:50000],targets[:50000])
testloader = DataLoader(testdata, batch_size = 128, shuffle = True)
net = MyNet()
torch.save(net.state_dict(), "model.pth")
net = net.to(device)
lossfn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr = 0.001)
epochs = 100


#Training with SGD
sgd_losses = []
sgd_accuracies = []
for epoch in range(epochs):
    net.train()
    epoch_losses = []
    epoch_accuracies = []
    for x,y in trainloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        optim.zero_grad()
        loss = lossfn(yhat,y)
        loss.backward()
        optim.step()
        epoch_losses.append(loss.item())
    sgd_losses.append(np.array(epoch_losses).mean())
    net.eval()
    for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        _, argmaxes = yhat.max(-1)
        correct = argmaxes == y
        epoch_accuracies.append(correct.sum().item())
    sgd_accuracies.append(np.array(epoch_accuracies).sum())
    print(f"Epoch({epoch}) : loss = {sgd_losses[-1]:.2f} || Accuracy = {sgd_accuracies[-1]/60000:.2f}")
torch.save(net.state_dict(), "final_SGD.pth")

#Training with Adam
net.load_state_dict(torch.load("model.pth"))
adam_losses = []
adam_accuracies = []
optim = torch.optim.Adam(net.parameters(), lr = 0.001)
for epoch in range(epochs):
    net.train()
    epoch_losses = []
    epoch_accuracies = []
    for x,y in trainloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        optim.zero_grad()
        loss = lossfn(yhat,y)
        loss.backward()
        optim.step()
        epoch_losses.append(loss.item())
    adam_losses.append(np.array(epoch_losses).mean())
    net.eval()
    for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        _, argmaxes = yhat.max(-1)
        correct = argmaxes == y
        epoch_accuracies.append(correct.sum().item())
    adam_accuracies.append(np.array(epoch_accuracies).sum())
    print(f"Epoch({epoch}) : loss = {adam_losses[-1]:.2f} || Accuracy = {adam_accuracies[-1]/60000:.2f}")
torch.save(net.state_dict(), "final_Adam.pth")

plt.figure(figsize = (20,5))
plt.subplot(121)
plt.title("Loss")
plt.plot(np.arange(epochs)+1, sgd_losses, label = "SGD Training loss")
plt.plot(np.arange(epochs)+1, adam_losses, label = "Adam Training loss")
plt.legend()
plt.subplot(122)
plt.title("Accuracy")
plt.plot(np.arange(epochs)+1, sgd_accuracies, label = "SGD Test Accuracy")
plt.plot(np.arange(epochs)+1, adam_accuracies, label = "Adam Test Accuracy")
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100/60000) for x in plt.gca().get_yticks()])
plt.legend()
plt.savefig("Epochs_Adam.png")


