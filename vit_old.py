# !pip - q
# install
# vit_pytorch
# linformer

# Import Libraries

from __future__ import print_function
import glob
from itertools import chain
import os
import random
import zipfile

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

print(f"Torch: {torch.__version__}")

# Training settings
batch_size = 64
epochs = 400
lr = 1e-4
gamma = 0.7
seed = 42

df_train = pd.read_csv('/truba/home/rdundar/ramiz/vit-chexpert/CheXpert-v1.0-small/train.csv')
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
    0: list(df_train_consolidation['Path']),
    1: list(df_train_atelectasis['Path']),
    2: list(df_train_pleural['Path']),
    3: list(df_train_edema['Path']),
    4: list(df_train_cardiomegaly['Path']),
}

train_list = []
labels_train = []

for key, val in train_cond_data.items():
    for l in val:
        train_list.append(l)
        labels_train.append(key)


test_cond_data = {
    0: list(df_test_consolidation['Path']),
    1: list(df_test_atelectasis['Path']),
    2: list(df_test_pleural['Path']),
    3: list(df_test_edema['Path']),
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


# df_train_2 = df_train[df_train['Cardiomegaly']==0]
# df_train = df_train[df_train['Cardiomegaly']==1.0]
# df_train = df_train.append(df_train_2)
# df_test_1 = df_test[df_test['Cardiomegaly']==1.0]
# df_test_2 = df_test[df_test['Cardiomegaly']==0.0]
# df_test = df_test_1.append(df_test_2)


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


train_dir = 'CheXpert-v1.0-small/train'
test_dir = 'CheXpert-v1.0-small/valid'

# train_list = list(df_train['Path'])
# test_list = list(df_test['Path'])

# print(f"Train Data: {len(train_list)}")
# print(f"Test Data: {len(test_list)}")


# print(len([i if i==1 for i in train_list]))

# labels_train = list(df_train['Cardiomegaly'])
# test_labels = list(df_test['Cardiomegaly'])

# cnt = 0 
# for e in labels_train:
#     if e == 1:
#         cnt += 1

# print(cnt/len(labels_train))

# cnt = 0 
# for e in test_labels:
#     if e == 1:
#         cnt += 1

# print(cnt/len(test_labels))


# Split

train_list, valid_list = train_test_split(train_list,
                                          test_size=0.2,
                                          stratify=labels_train,
                                          random_state=seed)

train_labels, valid_labels = train_test_split(labels_train,
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
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = int(self.label_list[idx])

        return img_transformed, label


train_data = CatsDogsDataset(train_list, train_labels, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, valid_labels, transform=test_transforms)
test_data = CatsDogsDataset(test_list, test_labels, transform=test_transforms)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

print(len(train_data), len(train_loader))

print(len(valid_data), len(valid_loader), flush=True)

# Effecient Attention
# Linformer

# efficient_transformer = Linformer(
#     dim=256,
#     seq_len=197,  # 7x7 patches + 1 cls-token
#     depth=6,
#     heads=8,
#     k=64
# )

# # Visual Transformer

# model = ViT(
#     dim=256,
#     image_size=224,
#     patch_size=16,
#     num_classes=5,
# #    transformer=efficient_transformer,  # nn.Transformer(d_model=128, nhead=8),
#     heads=8,
#     depth=6,
#     mlp_dim=1,
#     channels=1,
# ).to(device)


teacher = resnet50(pretrained=False)
teacher = teacher.cuda()
# teacher = ResNet(block=10, layers=10, num_classes=5)

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
    temperature=1,           # temperature of distillation
    alpha=0.6               # trade between main loss and distillation loss
).cuda()

# Training

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

best_acc = 0

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        # output = model(data)
        loss = distiller(data, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        v = model.to_vit().cuda()
        output = v(data)

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            
            v = model.to_vit().cuda()
            val_output = v(data)
            val_loss = distiller(data, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n", flush=True)

    torch.save(model.state_dict(), f'/truba/home/rdundar/ramiz/vit-chexpert/checkpoints/pretrained-net-{epoch + 1}.pt')
