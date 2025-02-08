from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class MalariaDataset(Dataset):
    def __init__(self, files, train= True):
        self.train = train
        self.train_transform  = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128),\
                                    transforms.CenterCrop(128), transforms.ColorJitter(brightness = (0.95,1.05),
                                    contrast = (0.95,1.05), saturation = (0.95,1.05), hue = 0.05),
                                    transforms.RandomAffine(5, translate = (0.01,0.1)), transforms.ToTensor()])
        self.test_transform  = transforms.Compose([transforms.ToPILImage(), transforms.Resize(128),\
                                    transforms.CenterCrop(128), transforms.ToTensor()])
        self.files = files
        self.name2class = {"Uninfected": 0, "Parasitized" : 1}

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, ix):
        clss = self.name2class[self.files[ix].split("\\")[-2]]
        img = cv2.imread(self.files[ix])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train:
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        return img,clss