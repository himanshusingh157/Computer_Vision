import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchsummary import summary
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

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
optimizer = torch.optim.SGD(model.parameters(), lr = 5e-3, momentum = 0.9, weight_decay = 5e-4)
epochs = 5

train_losses = []
val_losses = []
model.train()
for epoch in range(epochs):
    
    start_time = time.time()
    N = len(trainloader)
    train_epoch_losses = []
    for inputs in tqdm(trainloader):
        loss,losses = train_batch(inputs,model,optimizer)
        #locloss, regloss,objloss, rpnloss = [losses[k] for k in losses] 
        train_epoch_losses.append(loss.item())
    train_losses.append(np.array(train_epoch_losses).mean())
    N = len(valloader)
    val_epoch_losses = []
    for inputs in tqdm(valloader):
        loss,losses = val_batch(inputs,model)
        #locloss, regloss,objloss, rpnloss = [losses[k] for k in losses] 
        val_epoch_losses.append(loss.item())
    val_losses.append(np.array(val_epoch_losses).mean())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Epoch({epoch+1}) : Elapsed Time {elapsed_time:.0f} sec || Train loss = {train_losses[-1]:.2f} || ", end = "")
    print(f"Test loss = {val_losses[-1]:.2f}")
    torch.save(model.state_dict(), f"{epoch+1}_model.pth")
    if(epoch > 1):
        plt.plot(np.arange(epoch)+1, train_losses, label = "Training")
        plt.plot(np.arange(epoch)+1, val_losses, label = "Validation")

        plt.title('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig('Epochs.png')

    
#torch.save(model.state_dict(), "final_model.pth")


"""
##########  EVAL ########

def decode_faster_rcnn(output_):
    bbs = output_['boxes'].cpu().detach().numpy().astype(np.uint16)
    labels = np.array([target2label[i] for i in output_['labels'].cpu().detach().numpy()])
    confs = output_['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.05)
    bbs, confs, labels = [pred[ixs] for pred in [bbs, confs, labels]]
    if len(ixs) == 1:
        bbs = confs, labels = [np.array([pred]) for pred in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()

model.load_state_dict(torch.load("final_model.pth", weights_only=True))
model.eval()

for i, (images, targets) in enumarate(valloader):
    if i == 1:
        break
    images = [im for im in images]
    outputs = model(images)
    for j, out in enumarate(outputs):
        bbs, connfs, labels = decode_faster_rcnn(outputs)
        ####plot bounding box
        
"""
    