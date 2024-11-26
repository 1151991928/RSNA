import torch
import torchvision.models as models

# 加载预训练的FCN模型
fcn = models.segmentation.fcn_resnet101(pretrained=True)

# 修改最后的卷积层
fcn.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))

# 将Sigmoid激活函数添加到输出层
fcn.add_module('sigmoid', torch.nn.Sigmoid())

model=fcn
