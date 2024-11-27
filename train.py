import math
import numpy as np
import cv2 as cv
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader,Dataset
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
from dataset import dataset
from unet import UNet
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import datetime
from sklearn.metrics import jaccard_score
import torch
import torch.nn.functional as F

def compute_iou(pred, target, num_classes):
    """
    计算IoU

    参数:
    pred: Tensor[N, H, W] 预测分割结果
    target: Tensor[N, H, W] 真实标签
    num_classes: int 类别数量

    返回:
    iou: Tensor[num_classes] 每个类别的IoU值
    """
    ious = []
    pred = pred.argmax(dim=1)
    pred = F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
    target = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    for c in range(num_classes):
        # 计算交集
        intersection = torch.sum(pred[:, c, :, :] * target[:, c, :, :])
        # 计算并集
        union = torch.sum(pred[:, c, :, :]) + torch.sum(target[:, c, :, :]) - intersection
        # 计算IoU
        iou = intersection / (union + 1e-6)  # 防止除以0
        ious.append(iou)

    return torch.tensor(ious)

model=UNet(n_channels=1,n_classes=2)
model.cuda()
batch_size=8
epochs=20

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model.cuda()

criterion = torch.nn.CrossEntropyLoss()
criterion.cuda()

optimizer=torch.optim.Adam(model.parameters(), lr=0.00001)
tb_writer=SummaryWriter('./log')
total_train_step = 0
total_test_step = 0

for epoch in range(epochs):
    print(f'Epoch [{epoch+1}/{epochs}]')
    model.train()
    total_miou = 0.0
    train_bar = tqdm(train_dataloader)
    for data in train_bar:
        imgs, targets = data
        
        imgs=imgs.cuda()
        targets=targets.cuda()
        outputs = model(imgs)
        outputs = torch.softmax(outputs,dim=1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        

        train_bar.set_description(f'Train Step: {total_train_step} Loss: {loss.item():.4f}')

    model.eval()
    total_test_loss = 0
    total_miou = 0
    test_bar=tqdm(val_dataloader)
    with torch.no_grad():
        for data in test_bar:
            imgs, targets = data
            imgs=imgs.cuda()
            targets=targets.cuda()
            outputs = model(imgs)
            outputs = torch.softmax(outputs,dim=1)
            loss = criterion(outputs, targets)
            miou=compute_iou(outputs, targets, 2)
            total_test_loss += loss.item()
            total_miou += miou
    total_test_loss /= len(val_dataloader)
    total_miou /= len(val_dataloader)
    tb_writer.add_scalar('test_loss', total_test_loss, total_test_step)
    
    print(f'Test Loss: {total_test_loss}')
    print(f'Test miou: {total_miou}')
    total_test_step += 1
    torch.save(model.state_dict(), f'./model/model_{epoch}.pth')
    print('Model saved.')


