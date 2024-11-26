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
import matplotlib.pyplot as plt

classes = ['background','1']

# RGB color for each class
colormap = [[0,0,0],[255,255,255]]

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # 其他必要的预处理步骤
])
cm = np.array(colormap).astype('uint8')

def predict(model,path):
    img=Image.open(path).convert("RGB")
    img=transforms(img)
    img = img.unsqueeze(0).repeat(1, 1, 1, 1)
    output=model(img.cuda())
    pred = output.max(1)[1].squeeze().data.numpy()
    pred = cm[pred]
    return pred

state_dict=torch.load('./model.pth')
model.load_state_dict(state_dict)
a=predict(model,'C:/Users/armstrong/Desktop/sth/COVID_CXR/dataset/CXR/images/P_173.jpeg')
plt.imshow(a)
plt.axis('off')  # 关闭坐标轴
plt.show()


