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

import torchxrayvision as xrv

print(f"Torch: {torch.__version__}")

# Training settings
batch_size = 64
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


train_list, valid_list, train_labels, valid_labels= train_test_split(train_list, labels_train, # Fix
                                          test_size=0.2,
                                          stratify=labels_train,
                                          random_state=seed)


# train_list, valid_list = train_test_split(train_list,
#                                           test_size=0.2,
#                                           stratify=labels_train,
#                                           random_state=seed)

# train_labels, valid_labels = train_test_split(labels_train,
#                                               test_size=0.2,
#                                               stratify=labels_train,
#                                               random_state=seed)


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

    # def __getitem__(self, idx):
    #     img_path = self.file_list[idx]
    #     img = cv2.imread(img_path, 0)
    #     assert img.ndim == 2, "image must be gray image"
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #     img = Image.fromarray(img)
    #     img_transformed = self.transform(img)
        

    #     label = int(self.label_list[idx])

    #     return img_transformed, label

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('L') # single channel
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



# teacher = ResNet(block=10, layers=10, num_classes=5)


# teacher

def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == 'BatchNorm':
        return nn.BatchNorm2d(num_features, eps=eps)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(num_groups, num_features, eps=eps)
    elif norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(num_features, eps=eps,
                                 affine=True, track_running_stats=True)
    else:
        raise Exception('Unknown Norm Function : {}'.format(norm_type))

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']  # noqa


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',  # noqa
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',  # noqa
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',  # noqa
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',  # noqa
}

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 norm_type='Unknown'):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', get_norm(norm_type, num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', get_norm(norm_type, bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)  # noqa
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, norm_type='Unknown'):  # noqa
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, norm_type=norm_type)  # noqa
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,
                 norm_type='Unknown'):
        super(_Transition, self).__init__()
        self.add_module('norm', get_norm(norm_type, num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,  # noqa
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_  # noqa
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer  # noqa
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 norm_type='Unknown', num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):  # noqa

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),  # noqa
            ('norm0', get_norm(norm_type, num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, norm_type=norm_type,  # noqa
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)  # noqa
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, norm_type=norm_type)  # noqa
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', get_norm(norm_type, num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # out = self.classifier(out)
        return out


def densenet121(**kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_  # noqa
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),  # noqa
                     norm_type="BatchNorm", **kwargs)
    if True:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')  # noqa
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model


# teacher = densenet121()
# teacher = teacher.cuda()
# print(teacher)

teacher = xrv.models.DenseNet(weights="all")
print(teacher)

# teacher2 = resnet50(pretrained=False).cuda()
# print(teacher2)


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

for epoch in range(epochs):
    distiller.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        # output = model(data)
        optimizer.zero_grad()
        loss = distiller(data, label)

        loss.backward()
        optimizer.step()

        v = model.to_vit().cuda().eval()
        output = v(data)

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    
    distiller.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)
            
            v = model.to_vit().cuda().eval()
            val_output = v(data)
            val_loss = distiller(data, label)
            
            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(f"Epoch : {epoch + 1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n", flush=True)

    torch.save(model.state_dict(), f'/truba/home/rdundar/ramiz/vit-chexpert/checkpoints/pretrained-net-{epoch + 1}.pt')
