import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model():
    model = fasterrcnn_resnet50_fpn(pretarined = True)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat,3)
    return model


IMAGE_ROOT = "archive\\images\\images"
DF_RAW = pd.read_csv("archive\\df.csv")

train_ids, val_ids = train_test_split(DF_RAW.ImageID.unique(), test_size = 0.1)
train_df = DF_RAW[DF_RAW['ImageID'].isin(train_ids)]
val_df = DF_RAW[DF_RAW['ImageID'].isin(val_ids)]

traindataset = BTDataset(train_df,IMAGE_ROOT)
valdataset = BTDataset(val_df,IMAGE_ROOT)
trainloader =DataLoader(traindataset, batch_size = 4, collate_fn = traindataset.collate_fn)
valloader =DataLoader(valdataset, batch_size = 4, collate_fn = valdataset.collate_fn)

model = get_model().to(device)
rcnnlosses = ['loss_classifier','loss_box_reg','loss_objectness', 'loss_rpn_box_reg']
optimizer = torch.optim.SGD(model.parameters(), lr = 5e-3)
epochs = 1

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(epochs):
    start_time = time.time()
    N = len(trainloader)
    train_epoch_losses = []
    for ix, inputs in enumerate(trainloader):
        loss,losses = train_batch(inputs,model,optimizer)
        #locloss, regloss,objloss, rpnloss = [losses[k] for k in rcnnlosses] 
        train_epoch_losses.append(loss.item())
    N = len(valloader)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    