import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.models as models



if __name__ =='__main__':
    vgg19 = models.vgg19()
    img = Image.open('../git_code/pytorch-grad-cam-master/cam.jpg')
    tran = transforms.Compose([
        transforms.Resize(244,244),
        transforms.ToTensor(),
    ])
    input =tran(img)
    out = vgg19(input)
