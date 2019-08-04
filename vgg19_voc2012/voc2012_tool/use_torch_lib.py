# 使用torch自带的voc库函数示例
import os

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets.voc import VOCDetection


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
    image_set = 'train'  # 数据集类型为：train、trainval、val中的一种
    train = VOCDetection(root=root, year=year, image_set='train',
                         transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    train_loader = DataLoader(dataset=train, shuffle=True, batch_size=100, collate_fn=voc_collate_fn)
    trainval = VOCDetection(root=root, year=year, image_set='trainval',
                            transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    trainval_loader = DataLoader(dataset=trainval, shuffle=True, batch_size=100, collate_fn=voc_collate_fn)
    val = VOCDetection(root=root, year=year, image_set='val',
                       transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
    val_loader = DataLoader(dataset=val, shuffle=True, batch_size=100, collate_fn=voc_collate_fn)
    # for batch_idx, target in enumerate(train_loader):
    #     print(batch_idx, target)
    #     break
    train_images = train.images
    trainval_images = trainval.images
    val_images = val.images

    # 测试是否所有图片都被使用到
    annotations_file = os.listdir(r'G:\大二\实验室学习\VOCdevkit\VOC2012\Annotations')
    annotations = [i[:11] for i in annotations_file]
    used_annotations = [j[-15:-4] for j in train_images]
    for i in [j[-15:-4] for j in train_images]:
        used_annotations.append(i)
    for i in [j[-15:-4] for j in trainval_images]:
        used_annotations.append(i)
    for i in [j[-15:-4] for j in val_images]:
        used_annotations.append(i)
    used_annotations_set = set(used_annotations)

    annotations = set(annotations)
    print(annotations == used_annotations)
