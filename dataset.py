import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]

cm2lbl = np.zeros(256**3) # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i # 建立索引

def image2label(img):
    data = np.array(img, dtype='int32')
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    return np.array(cm2lbl[idx], dtype='int64') # 根据索引得到 label 矩阵

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
        img = Image.open(self.data_path+'/'+self.data[index])
        mask = Image.open(self.mask_path+'/'+self.mask[index]).convert('RGB')
        
        img = self.transform(img)
        mask = self.transform(mask)
        mask = image2label(mask)
        return img,mask
