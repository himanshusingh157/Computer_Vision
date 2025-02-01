import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.X = torch.tensor(x).float()
        self.Y = torch.tensor(y).float()

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,index):
        return self.X[index], self.Y[index]

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10,5)
        self.layer2 = nn.Linear(5,1)
        self.actiavtionfn = nn.ReLU()
        
    def forward(self,x):
        y = self.actiavtionfn(self.layer1(x))
        y = self.layer2(y)
        return y
        

x = np.random.rand(400,10)
y = np.random.randint(0,2,(400,1))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device Available : {device}")

mydata = MyDataset(x,y)
dataloader = DataLoader(mydata, batch_size = 40, shuffle = True)
net = MyNet().cuda()
mse_loss = nn.MSELoss()
optim = torch.optim.SGD(net.parameters(), lr = 0.0001)

for epoch in range(10):
    print(f"Epoch({epoch}) : ", end = "")
    for x,y in dataloader:
        x = x.to(device)
        y = y.to(device)
        yhat = net(x)
        optim.zero_grad()
        loss = mse_loss(yhat,y)
        print(f"{loss.item()} ", end = "")
        loss.backward()
        optim.step()
    print()


    