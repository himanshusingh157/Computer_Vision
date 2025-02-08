import torch
import os
import random
from glob import glob
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import time
from model import *
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


root_folder = os.path.join(os.getcwd(), "archive", "cell_images")
all_files = glob(f"{root_folder}\\*\\*.png")
random.shuffle(all_files)
train_images_count = (8*len(all_files))//10
train_files = all_files[:train_images_count]
test_files = all_files[train_images_count:]

traindata = MalariaDataset(train_files, train=True)
trainloader = DataLoader(traindata, batch_size = 64, shuffle = True)
testdata = MalariaDataset(test_files, train = False)
testloader = DataLoader(testdata, batch_size = 64, shuffle = True)

model = MalariaClassifier().to(device)
#summary(model, (3,128,128))
lossfn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
epochs = 50


train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(epochs):
    start_time = time.time()
    #Train
    model.train()
    train_epoch_losses = []
    train_epoch_accuracies = []
    for x,y in trainloader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        yhat = model(x)
        _, argmaxes = yhat.max(-1)
        correct = argmaxes == y
        train_epoch_accuracies.append(correct.sum().item())
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
            _, argmaxes = yhat.max(-1)
            correct = argmaxes == y
            test_epoch_accuracies.append(correct.sum().item())
    test_accuracies.append(np.array(test_epoch_accuracies).sum())
    test_losses.append(np.array(test_epoch_losses).mean())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Epoch({epoch+1}) : Elapsed Time {elapsed_time:.0f} sec || Train loss = {train_losses[-1]:.2f} || ", end = "")
    print(f"Train Accuracy = {train_accuracies[-1]/len(traindata):.2f} || Test loss = {test_losses[-1]:.2f} || ", end = "")
    print(f"Test Accuracy = {test_accuracies[-1]/len(testdata):.2f} ")
    
torch.save(model.state_dict(), "final_model.pth")


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