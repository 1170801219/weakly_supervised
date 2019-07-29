import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader

from voc2012_tool.voc2012_dataset import voc2012_dataset, Rescale, ToTensor,label_int_map

# 模型显存占用监测函数
# model：输入的模型
# input：实际中需要输入的Tensor变量
# type_size 默认为 4 默认类型为 float32

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums


    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))


class my_VGG(nn.Module):
    def __init__(self,model,class_num):
        super(my_VGG,self).__init__()
        self.vgg = model
        self.soft_max = nn.Softmax()
        self.fc = nn.Linear(1000,class_num)

    def forward(self, x):
        out = self.vgg(x)
        out = self.soft_max(out)
        out = self.fc(out)
        return out

if __name__ =='__main__':
    verbose = True
    label_int_map = label_int_map
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    vgg_model = my_VGG(models.vgg11(), len(label_int_map))
    if verbose:
        print('加载模型之前的显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())
    vgg_model.to(device)
    if verbose:
        print('加载模型之后的显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())

    batch_size,num_works,shuffle,lr = 2,6,True,0.01
    jpeg_images = 'G:/大二/实验室学习/voc2012/VOC2012/JPEGImages'
    main_dir = 'G:/大二/实验室学习/voc2012/VOC2012/ImageSets/Main'
    data_type = 'train'
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg_model.parameters(), lr=lr)

    dataset_train = voc2012_dataset(data_type,
                                    main_dir,
                                    jpeg_images,
                                    label_int_map,
                                    transform=transforms.Compose([Rescale((224,224)),ToTensor()]))
    dataloader_train = DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=shuffle,num_workers=num_works)

    if verbose:
        print('使用计算平台',device)
        print('batch size:',batch_size)
    for batch,sample_batched in enumerate(dataloader_train):
        print('第',batch,'个batch')
        torch.cuda.empty_cache()

        print('开始计算，清除catch之后的显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())
        sample_batched['image'] = sample_batched['image'].to(device)
        print('读取一个batch_img,显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())
        sample_batched['label'] = sample_batched['label'].to(device)
        print('读取一个batch_label,显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())

        # 前向传播
        pred = vgg_model(sample_batched['image'].float())
        print('前向传播,显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())

        # 计算loss
        loss = criterion(pred,sample_batched['label'])
        print('计算loss,显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())


        if verbose:
            print('loss: ',loss.item())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        print('反向传播,显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())

        # 更新参数
        optimizer.step()
        print('更新参数,显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())
