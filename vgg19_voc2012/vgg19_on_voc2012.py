import csv
import sys

import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets.voc import VOCDetection
from torchvision.models.vgg import make_layers, cfgs

_output = sys.stdout
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def voc_collate_fn(batch_list):
    '''
    将数据打包成一个batch来训练
    '''
    images = default_collate([batch_list[i][0] for i in range(len(batch_list))])
    annotations = {}
    for k in batch_list[0][1]['annotation']:
        annotations[k] = [batch_list[i][1]['annotation'][k] for i in range(len(batch_list))]
    object_list = []
    for i in annotations['object']:
        if type(i) == list:
            object_list.append(i)
        else:
            l = []
            l.append(i)
            object_list.append(l)
    annotations['object'] = object_list
    return {'images': images, 'annotations': annotations}


class my_VGG(models.VGG):
    vgg_cfg_map = {11:'A',13:'B',16:'D',19:'E'}

    def __init__(self, num_classes, vgg_type, pre_train, batch_norm=False, verbose=False):
        self.verbose = verbose
        cfg=self.vgg_cfg_map[vgg_type]
        super(my_VGG,self).__init__(make_layers(cfgs[cfg], batch_norm=batch_norm), num_classes=num_classes)
        if pre_train:
            state_dict = load_state_dict_from_url(model_urls['vgg{}'.format(vgg_type)],
                                                  progress=True)
            self.load_state_dict(state_dict)


def forward(self, x):
        if self.verbose:
            print('输入数据维度：{}'.format(x.shape))
        return super(my_VGG, self).forward(x)

def gen_BECloss_label(objects, label_int_map):
    '''
    利用一个batch中的annotations生成符合BECloss函数的target标签
    :param objects:
    :return:
    '''
    result = []
    for lst in objects:
        label = [0] * len(label_int_map)
        for d in lst:
            assert d['name'] in label_int_map
            label[label_int_map[d['name']]] = 1
        result.append(label)
    return torch.tensor(result).float()


def training(loader, vgg_model, criterion, optimizer, output_processor, label_int_map, verbose, device, use_gpu, ep):
    sum_loss, avg_loss = 0, 0
    for batch_idx, sample_batched in enumerate(loader):

        if use_gpu:
            torch.cuda.empty_cache()

        sample_batched['images'] = sample_batched['images'].to(device)

        label = gen_BECloss_label(sample_batched['annotations']['object'], label_int_map)
        label = label.to(device)

        # 前向传播
        out = vgg_model(sample_batched['images'].float())

        # 计算loss
        loss = criterion(output_processor(out), label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        sum_loss += loss.item()
        avg_loss = sum_loss / (batch_idx + 1)
        if verbose:
            _output.write(
                ('\rtrain\t[epoch:{2}]\t[batch:{0}]\t[loss:{1}]\t[average loss:{3}]').format(batch_idx, loss.item(),
                                                                                             ep + 1, avg_loss))
    print('\r')
    return avg_loss


def validating(loader, vgg_model, criterion, optimizer, output_processor, label_int_map, verbose, device, use_gpu, ep):
    sum_loss, avg_loss = 0, 0
    for batch_idx, sample_batched in enumerate(loader):

        if use_gpu:
            torch.cuda.empty_cache()

        sample_batched['images'] = sample_batched['images'].to(device)

        label = gen_BECloss_label(sample_batched['annotations']['object'], label_int_map)
        label = label.to(device)

        # 前向传播
        with torch.no_grad():
            out = vgg_model(sample_batched['images'].float())

        # 计算loss
        loss = criterion(output_processor(out), label)
        sum_loss += loss.item()
        avg_loss = sum_loss / (batch_idx + 1)
        if verbose:
            _output.write(
                ('\rvalid\t[epoch:{2}]\t[batch:{0}]\t[loss:{1}]\t[average loss:{3}]').format(batch_idx, loss.item(),
                                                                                             ep + 1, avg_loss))
    print('\r')
    return avg_loss


def train_model(vgg_type, voc2012_root, batch_size, epoch, lr, log_save_file, use_gpu=True, shuffle=True, verbose=True,
                num_works=0, prev_model=None, pre_train=False):
    table_head = ['epoch', 'train loss', 'validate loss']
    csv_writer = csv.writer(open(log_save_file, 'w', newline=''))
    csv_writer.writerow(table_head)
    label_int_map = {'bus': 0, 'bird': 1, 'dog': 2, 'sofa': 3, 'cow': 4, 'tvmonitor': 5, 'person': 6, 'bicycle': 7,
                     'motorbike': 8, 'diningtable': 9, 'bottle': 10, 'chair': 11, 'boat': 12, 'car': 13, 'cat': 14,
                     'sheep': 15, 'train': 16, 'pottedplant': 17, 'aeroplane': 18, 'horse': 19}
    vgg_model = my_VGG(num_classes=len(label_int_map), vgg_type=vgg_type, pre_train=pre_train, batch_norm=False,
                       verbose=False)
    # 载入以前训练的模型
    if prev_model != None:
        vgg_model = (torch.load(prev_model))
        vgg_model.eval()
        if verbose:
            print('载入之前模型：{}'.format(prev_model))
    use_gpu = torch.cuda.is_available() and use_gpu
    device = torch.device('cuda' if use_gpu else 'cpu')
    if use_gpu:
        if verbose:
            print('使用{}块GPU进行训练'.format(torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            vgg_model = nn.DataParallel(vgg_model)
    vgg_model.to(device)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    criterion = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.SGD(vgg_model.parameters(), lr=lr)
    sigmoid = nn.Sigmoid()

    # 读取数据集
    voc2012_train = VOCDetection(root=voc2012_root, year='2012', image_set='train', transform=transform_train)
    dataloader_train = DataLoader(dataset=voc2012_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_works,
                                  collate_fn=voc_collate_fn)
    voc2012_val = VOCDetection(root=voc2012_root, year='2012', image_set='trainval', transform=transform_train)
    dataloader_val = DataLoader(dataset=voc2012_val, batch_size=batch_size, shuffle=shuffle, num_workers=num_works,
                                collate_fn=voc_collate_fn)

    if verbose:
        print('数据初始化完毕，开始训练')
        print('使用计算平台', device)
        print('batch size:', batch_size)
    val_loss = float('inf')

    # 开始模型训练
    for ep in range(epoch):
        tl = training(dataloader_train, vgg_model, criterion, optimizer, sigmoid, label_int_map, verbose, device,
                      use_gpu, ep)
        vl = validating(dataloader_val, vgg_model, criterion, optimizer, sigmoid, label_int_map, verbose, device,
                        use_gpu, ep)

        # 记录此次训练结果
        epoch_result = [ep, tl, vl]
        csv_writer.writerow(epoch_result)
        # 保存loss下降的模型
        if vl < val_loss:
            if verbose:
                print('new validation loss:{}'.format(vl))
                print('保存最优训练结果： vgg_{1}_epoch_{0}_dict.pth'.format(ep, vgg_type))
            torch.save(vgg_model.state_dict(), 'vgg_{1}_epoch_{0}_dict.pth'.format(ep, vgg_type))
            val_loss = vl
        # 保存最近一次训练出的模型
        if verbose:
            print('保存此次训练结果：[epoch:{}]\t[train loss:{}]\t[valid loss:{}]'.format(ep + 1, tl, vl))
        torch.save(vgg_model, 'vgg_{}_trained.pth'.format(vgg_type))
    print('模型vgg_{}训练完毕'.format(vgg_type))

if __name__ =='__main__':
    # 初始化参数
    verbose = True
    use_gpu = True
    batch_size, num_works, shuffle, lr = 16, 0, True, 0.01
    epoch = 300
    vgg_type = 19  # 11，13，16，19 中选择一个
    voc2012_root = r'~/dataset'
    log_save_file = 'vgg_{}_train_log.csv'.format(vgg_type)

    #     训练模型
    train_model(vgg_type, voc2012_root, batch_size, epoch, lr, log_save_file, use_gpu,
                shuffle, verbose, num_works, pre_train=False)  # 'vgg_{}_trained.pth'.format(vgg_type)
