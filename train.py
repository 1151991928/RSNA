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
from dataset import my_dataset
from fcn import model
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import datetime
from sklearn.metrics import jaccard_score

classes = ['background','1']
# RGB color for each class
colormap = [[0,0,0],[255,255,255]]

if torch.cuda.is_available():
    device = torch.device('cuda')

batch_size=64
epochs=20
path='./dataset/rsna'

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # 其他必要的预处理步骤
])

dataset=my_dataset(path,transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model=model()
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
criterion.to(device)

optimizer=torch.optim.Adam(model.parameters(), lr=0.0001)
tb_writer=SummaryWriter('./log')
total_train_step = 0
total_test_step = 0

for epoch in range(epochs):
    print(f'Epoch [{epoch+1}/{epochs}]')
    model.train()
    total_miou = 0.0
    for data in tqdm(train_dataloader):
        imgs, targets = data
        imgs=imgs.to(device)
        targets=targets.to(device)
        outputs = model(imgs)
        outputs = F.log_softmax(outputs, dim=1)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        

        if total_train_step % 100 == 0:
            tb_writer.add_scalar('train_loss', loss.item(), total_train_step)
            print(f'Train Loss: {loss.item()}')

    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in tqdm(val_dataloader):
            imgs, targets = data
            imgs=imgs.to(device)
            targets=targets.to(device)
            outputs = model(imgs)
            outputs = F.log_softmax(outputs, dim=1)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1)
            for output, target in zip(outputs, targets):
                miou=jaccard_score(target.numpy(),output.numpy(), average='macro')
                total_miou += miou
    miou = total_miou / len(val_dataloader)
    total_test_loss /= len(val_dataloader)
    tb_writer.add_scalar('test_loss', total_test_loss, total_test_step)
    tb_writer.add_scalar('test_miou', miou, total_test_step)
    print(f'Test Loss: {total_test_loss}')
    print(f'Test miou: {miou}')
    total_test_step += 1
    torch.save(model.state_dict(), f'./model/model_{epoch}.pth')
    print('Model saved.')


