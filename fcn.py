import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
# 加载预训练的FCN模型
model = models.segmentation.fcn_resnet50(pretrained=True)

# 修改最后的卷积层
model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))

# 将Sigmoid激活函数添加到输出层
model.add_module('sigmoid', torch.nn.Sigmoid())

if __name__ == '__main__':
    img=torch.randn(1, 3, 512, 512)
    output=model(img)
    print(output)
