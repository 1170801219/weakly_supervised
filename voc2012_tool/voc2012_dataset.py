#运行需要已经安装：
# scikit-image: For image io and transforms
# pandas: For easier csv parsing

from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings

from voc2012_tool.dataset_voc2012_utils import get_files_absolute_path, parse_label_image

warnings.filterwarnings("ignore")
plt.ion()   # interactive mode

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
            例： re=Rescale(100),输入图片大小为(200,400),输出图片大小为(100,200)
                re=Rescale((100,100)),输入图片大小为(200,400),输出图片大小为(100,100)
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))


        return {'image': img, 'label': sample['label']}

class RandomCrop(object):
    """Crop randomly the image in a sample.
        随机裁剪图片，可用于数据增强

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]


        return {'image': image, 'label': sample['label']}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image= sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': sample['label']}

class voc2012_dataset(Dataset):
    '''voc2012 dataset'''
    def __init__(self,type,main_dir,image_dir,label_map,transform=None,verbose=False):
        self.transform = transform
        self.imgs=[]
        self.label_num = len(label_map)
        if image_dir[-1]!='/':
            image_dir=image_dir+'/'
        self.image_dir=image_dir
        files = get_files_absolute_path(main_dir)
        if type not in ['train','val','trainval']:
            raise RuntimeError('输入的数据类型错误，可接受类型：train,val,trainval')

        #将所有的图片加入到数据集中
        for file in files:
            label_image_dict,value_dict=parse_label_image(file,verbose=verbose)
            label = file.split('/')[-1].split('_')[0]
            data_type = file.split('/')[-1].split('_')[1].split('.')[0]
            if data_type == type:
                for img in value_dict[1]:
                    self.imgs.append((img,label_map[label]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_path=os.path.join(self.image_dir,self.imgs[index][0]+'.jpg')
        image = io.imread(image_path)
        sample ={'image':image,
                 'label':self.imgs[index][1]}
        if self.transform:
            sample=self.transform(sample)
        return sample


label_int_map ={'bus': 0,'bird': 1,'dog': 2,'sofa': 3,'cow': 4,'tvmonitor': 5,'person': 6,'bicycle': 7,'motorbike': 8,'diningtable': 9,'bottle': 10,'chair': 11,'boat': 12,'car': 13,'cat': 14,'sheep': 15,'train': 16,'pottedplant': 17,'aeroplane': 18,'horse': 19}

if __name__ =='__main__':
    # 示例代码
    int_label_map = {}
    for l in label_int_map:
        int_label_map[label_int_map[l]]=l
    dataset = voc2012_dataset('train',
                              'G:/大二/实验室学习/voc2012/VOC2012/ImageSets/Main',
                              'G:/大二/实验室学习/voc2012/VOC2012/JPEGImages',
                              label_int_map,
                              transform=transforms.Compose([Rescale((224,224))]))

    s = dataset[0]

    # 展示图片
    fig = plt.figure()

    for i in range(5):
        sample = dataset[i*80]
        print('label:' + int_label_map[sample['label']],
              'shape' + str(sample['image'].shape))
        plt.imshow(sample['image'])
        plt.pause(0.01)
        plt.show()
