import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch



class my_dataset(Dataset):
    def __init__(self,path,transform):
        self.path = path
        self.data_path = f'{path}/CXR'
        self.mask_path = f'{path}/Mask'
        self.data = os.listdir(self.data_path)
        self.mask = os.listdir(self.mask_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = Image.open(self.data_path+'/'+self.data[index]).convert('L')
        mask = Image.open(self.mask_path+'/'+self.mask[index]).convert('L')
        
        img = self.transform(img)
        mask = self.transform(mask)
        
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        return img,mask
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # 其他必要的预处理步骤
])
path='./dataset/rsna'
dataset=my_dataset(path,transform)
