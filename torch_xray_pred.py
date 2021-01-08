# !pip - q
# install
# vit_pytorch
# linformer

# Import Libraries

from __future__ import print_function
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import glob
from itertools import chain
import os
import random
import cv2
import zipfile
import torch.utils.model_zoo as model_zoo
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
from vit_pytorch.efficient import ViT
from torchvision.models import resnet50
from vit_pytorch.distill import DistillableViT, DistillWrapper
import sklearn, sklearn.metrics
import torchxrayvision as xrv
import sklearn, sklearn.metrics

print(f"Torch: {torch.__version__}")

# Training settings
batch_size = 32
epochs = 400
lr = 1e-4
gamma = 0.7
seed = 42

df_train = pd.read_csv('/truba/home/rdundar/ramiz/vit-chexpert/CheXpert-v1.0-small/train-small.csv')
df_test = pd.read_csv('/truba/home/rdundar/ramiz/vit-chexpert/CheXpert-v1.0-small/valid.csv')

df_train = df_train[df_train['Frontal/Lateral'] == 'Frontal']
df_train = df_train[df_train['AP/PA'] == 'AP']
df_train_cardiomegaly = df_train[((df_train['Cardiomegaly'] == 1.0) & (df_train['Consolidation'] != 1.0) & (df_train['Edema'] != 1.0) & (df_train['Pleural Effusion'] != 1.0) & (df_train['Atelectasis'] != 1.0))]
df_train_consolidation = df_train[((df_train['Consolidation'] == 1.0) & (df_train['Cardiomegaly'] != 1.0) & (df_train['Edema'] != 1.0) & (df_train['Cardiomegaly'] != 1.0) & (df_train['Atelectasis'] != 1.0))]
df_train_edema = df_train[((df_train['Edema'] == 1.0) & (df_train['Consolidation'] != 1.0) & (df_train['Cardiomegaly'] != 1.0) & (df_train['Cardiomegaly'] != 1.0) & (df_train['Atelectasis'] != 1.0))]
df_train_pleural = df_train[((df_train['Pleural Effusion'] == 1.0) & (df_train['Consolidation'] != 1.0) & (df_train['Edema'] != 1.0) & (df_train['Cardiomegaly'] != 1.0) & (df_train['Atelectasis'] != 1.0))]
df_train_atelectasis = df_train[((df_train['Atelectasis'] == 1.0) & (df_train['Consolidation'] != 1.0) & (df_train['Edema'] != 1.0) & (df_train['Pleural Effusion'] != 1.0) & (df_train['Cardiomegaly'] != 1.0))]

df_test = df_test[df_test['Frontal/Lateral'] == 'Frontal']
df_test = df_test[df_test['AP/PA'] == 'AP']
df_test_cardiomegaly = df_test[((df_test['Cardiomegaly'] == 1.0) & (df_test['Consolidation'] != 1.0) & (df_test['Edema'] != 1.0) & (df_test['Pleural Effusion'] != 1.0) & (df_test['Atelectasis'] != 1.0))]
df_test_consolidation = df_test[((df_test['Consolidation'] == 1.0) & (df_test['Cardiomegaly'] != 1.0) & (df_test['Edema'] != 1.0) & (df_test['Pleural Effusion'] != 1.0) & (df_test['Atelectasis'] != 1.0))]
df_test_edema = df_test[((df_test['Edema'] == 1.0) & (df_test['Consolidation'] != 1.0) & (df_test['Pleural Effusion'] != 1.0) & (df_test['Cardiomegaly'] != 1.0) & (df_test['Atelectasis'] != 1.0))]
df_test_pleural = df_test[((df_test['Pleural Effusion'] == 1.0) & (df_test['Consolidation'] != 1.0) & (df_test['Edema'] != 1.0) & (df_test['Cardiomegaly'] != 1.0) & (df_test['Atelectasis'] != 1.0))]
df_test_atelectasis = df_test[((df_test['Atelectasis'] == 1.0) & (df_test['Consolidation'] != 1.0) & (df_test['Edema'] != 1.0) & (df_test['Pleural Effusion'] != 1.0) & (df_test['Cardiomegaly'] != 1.0))]

train_cond_data = {
    0: list(df_train_atelectasis['Path']),
    1: list(df_train_consolidation['Path']),
    2: list(df_train_edema['Path']),
    3: list(df_train_pleural['Path']),
    4: list(df_train_cardiomegaly['Path']),
}

train_list = []
labels_train = []
# 0 1 4 9 10
for key, val in train_cond_data.items():
    for l in val:
        train_list.append(l)
        labels_train.append(key)


test_cond_data = {
    0: list(df_test_atelectasis['Path']),
    1: list(df_test_consolidation['Path']),
    2: list(df_test_edema['Path']),
    3: list(df_test_pleural['Path']),
    4: list(df_test_cardiomegaly['Path']),
}

test_list = []
test_labels = []

for key, val in test_cond_data.items():
    for l in val:
        test_list.append(l)
        test_labels.append(key)


del df_test
del df_train
del df_train_consolidation
del df_train_atelectasis
del df_train_pleural
del df_train_edema
del df_train_cardiomegaly


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

device = 'cuda'

# Load Data

train_list, valid_list, train_labels, valid_labels= train_test_split(train_list, labels_train, # Fix
                                          test_size=0.2,
                                          stratify=labels_train,
                                          random_state=seed)


print(f"Train Data: {len(train_list)}")
print(f"Train Labels: {len(train_labels)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Validation Data: {len(valid_labels)}")
print(f"Test Data: {len(test_list)}")
print(f"Test Labels: {len(test_labels)}")


# Image Augumentation

train_transforms = transforms.Compose(
    [
        xrv.datasets.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


# Load Datasets

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, label_list, transform=None):
        self.file_list = file_list
        self.transform = transform
        self.label_list = label_list

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('L') # single channel
        img_transformed = self.transform(img)

        label = int(self.label_list[idx])

        #print('img_transformed', img_transformed[:10])

        return img_transformed, label




train_data = CatsDogsDataset(train_list, train_labels, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, valid_labels, transform=test_transforms)
test_data = CatsDogsDataset(test_list, test_labels, transform=test_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))

print(len(valid_data), len(valid_loader), flush=True)

teacher = xrv.models.DenseNet(weights="all").eval() ################################
print(teacher)

model = DistillableViT(
    image_size=256,
    patch_size=16,
    num_classes=5,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1,
    channels=1,
)

distiller = DistillWrapper(
    student=model,
    teacher=teacher,
    temperature=3,           # temperature of distillation
    alpha=0.5               # trade between main loss and distillation loss
).cuda()

# Training

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=25, gamma=gamma)

best_acc = 0

# selected = [0, 1, 4, 9, 10]
# teacher_logits = teacher_logits[:, selected]

d_chex = xrv.datasets.CheX_Dataset(imgpath="/truba/home/rdundar/ramiz/vit-chexpert/CheXpert-v1.0-small",
                                   csvpath="/truba/home/rdundar/ramiz/vit-chexpert/CheXpert-v1.0-small/valid.csv",
                                   transform=train_transforms,
                                   views=["PA","AP"], unique_patients=False)

# d_nih = xrv.datasets.NIH_Dataset(imgpath="/lustre04/scratch/cohenjos/NIH/images-224")
sample = d_chex[40]

print('sample shape', sample["img"].shape)
print('unsqueeze shape', sample["img"].unsqueeze(0).shape)

print(d_chex)
print(sample.keys())


with torch.no_grad():
    # for epoch in range(1):
    #     epoch_accuracy = 0

    #     for data, label in train_loader:
    #         data = data.to(device)
    #         label = label.to(device)
    #         print('data', data)
    #         output = teacher(data)
    #         print('data shape', data.shape)
    #         print('output', output)


    #         selected = [0, 1, 4, 9, 10]
    #         output = output[:, selected]

    #         acc = (output.argmax(dim=1) == label).float().mean()

    #         epoch_accuracy += acc / len(train_loader)

            
    #         print('label', label)
            

        


    #     print(f"Epoch : {epoch + 1} - acc: {epoch_accuracy:.4f}\n", flush=True)
        
    epoch_accuracy = 0

    # for i in range(50):
    #     im = torch.from_numpy(d_chex[i]["img"]).unsqueeze(0).cuda()
    #     print(im.shape)
    #     output = teacher(im)
    #     print(output)
    #     print(torch.from_numpy(d_chex[i]["lab"]))
    #     acc = (output.argmax(dim=1) == torch.from_numpy(d_chex[i]["lab"]).cuda()).float().mean()
    #     epoch_accuracy += acc / len(train_loader)

    # print(f"Epoch : {epoch + 1} - acc: {epoch_accuracy:.4f}\n", flush=True)

    outs = []
    labs = []
    for i in np.random.randint(0,len(d_chex),100):
        sample = d_chex[i]
        labs.append(sample["lab"])
        out = teacher(sample["img"].unsqueeze(0).cuda()).cuda()
        out = torch.sigmoid(out)
        outs.append(out.detach().cpu().numpy()[0])

    
    for i in range(14):
        if len(np.unique(np.asarray(labs)[:,i])) > 1:
            auc = sklearn.metrics.roc_auc_score(np.asarray(labs)[:,i], np.asarray(outs)[:,i])
        else:
            auc = "(Only one class observed)"
        print(xrv.datasets.default_pathologies[i], auc)

