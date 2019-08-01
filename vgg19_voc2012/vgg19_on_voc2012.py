import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models.vgg import make_layers, cfgs

from voc2012_tool.voc2012_dataset import voc2012_dataset, Rescale, ToTensor, label_int_map, show_image



class my_VGG(models.VGG):
    vgg_cfg_map = {11:'A',13:'B',16:'D',19:'E'}
    def __init__(self,num_classes,vgg_type,batch_norm=False):
        cfg=self.vgg_cfg_map[vgg_type]
        super(my_VGG,self).__init__(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=num_classes)



def test():
    jpeg_images = 'G:/VOCdevkit/VOC2012/JPEGImages'
    main_dir = 'G:/VOCdevkit/VOC2012/ImageSets/Main'
    data_type = 'train'
    model = torch.load(r'C:\Users\1170300519\Desktop\YANGJIN\experimental_code\vgg19_voc2012\vgg_19_trained.pth')
    dataset_train = voc2012_dataset(data_type,
                                    main_dir,
                                    jpeg_images,
                                    label_int_map,
                                    transform=transforms.Compose([Rescale((224,224))]))

    input = dataset_train[0]
    show_image(input)
    to_tentor=ToTensor()
    in_=to_tentor(input)['image'].unsqueeze(dim=0).to(torch.device('cuda')).float()
    out = model(in_)
    print('输出',out)
    print('输入',input['label'])



if __name__ =='__main__':
    # test()

    verbose = False
    label_int_map = label_int_map
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    vgg_model = my_VGG(num_classes=len(label_int_map),vgg_type=19,batch_norm=False)
    if verbose:
        print('加载模型之前的显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())
    vgg_model.to(device)
    if verbose:
        print('加载模型之后的显存占用',torch.cuda.memory_allocated(),'/',torch.cuda.max_memory_allocated())

    torch.save(vgg_model,'vgg19_raw.pt')

    batch_size,num_works,shuffle,lr = 16,6,True,0.01
    epoch = 30
    jpeg_images = r'G:\大二\实验室学习\VOCdevkit\VOC2012\JPEGImages'
    main_dir = r'G:\大二\实验室学习\VOCdevkit\VOC2012\ImageSets\Main'
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
    index=0
    for ep in range(epoch):
        if ep%3 ==0:
            torch.save(vgg_model,r'G:\my_code\python\weakly_superivsed\experimental_code\model\vgg_19_epoch_'+str(ep)+'.pth')
        print('epoch',ep+1)
        for batch,sample_batched in enumerate(dataloader_train):
            index+=1

            # print('第',batch,'个batch')
            torch.cuda.empty_cache()

            sample_batched['image'] = sample_batched['image'].to(device)
            sample_batched['label'] = sample_batched['label'].to(device)

            # 前向传播
            pred = vgg_model(sample_batched['image'].float())

            # 计算loss
            loss = criterion(pred,sample_batched['label'])



            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 更新参数
            optimizer.step()
            if index %10 ==0:
                print('计算次数',index)
                if verbose:
                    print('loss: ',loss.item())
        torch.save(vgg_model, 'vgg_19_trained.pth')
