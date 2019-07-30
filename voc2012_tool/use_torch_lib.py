# 使用torch自带的voc库函数示例
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import DataLoader
from torchvision import transforms
import torch



def voc_collate_fn(batch_list):
    print(batch_list)
    images = default_collate([batch_list[i][0] for i in range(len(batch_list))])
    annotations = {}
    for k in batch_list[0][1]['annotation']:
        annotations[k] = [batch_list[i][1]['annotation'][k] for i in range(len(batch_list))]
    object_list = []
    for i in annotations['object']:
        if type(i)==list:
            object_list.append(i)
        else:
            l = []
            l.append(i)
            object_list.append(l)
    annotations['object'] = object_list
    return {'images':images,'annotations':annotations}


if __name__ =='__main__':
    root= 'G:/大二/实验室学习' # 文件夹下面包含VOCdevkit/VOC2012
    year= '2012'
    image_set = 'trainval' # 数据集类型为：train、trainval、val中的一种
    voc2012 = VOCDetection(root=root,year=year,image_set=image_set,transform =transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]))
    voc2012_loader = DataLoader(dataset=voc2012,shuffle=True,batch_size=100,collate_fn=voc_collate_fn)
    for batch_idx, target in enumerate(voc2012_loader):
        print(batch_idx, target)
        break

