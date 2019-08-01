import torch
from torch.autograd import Variable
from torchvision import models
import torch.optim as optim
import numpy as np
import torchvision
import time
from PIL import Image
from torchvision.utils import save_image
from torchvision import datasets, models, transforms
import  torch.nn as nn
import torch.nn.functional as F
import os
import math
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import average_precision_score
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs=30
batch_size=96
input_size=224
DATA_PATH = '/media/data/users/master/2018/zhanghan/data/voc2012/{}'
LABEL_PATH = '/media/data/users/master/2018/zhanghan/data/voc2012/{}_labels.txt'

class VOCdataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.labels = np.loadtxt(label_path, dtype=np.float32)[:, 1:]
        with open(label_path, 'r') as f:
            self.img_name_list = [line.split()[0] + '.jpg' for line in f]

    def __getitem__(self, index):
        img_pil = Image.open(os.path.join(
            self.data_path, self.img_name_list[index]))
        img_tensor = self.transform(img_pil)
        label = torch.from_numpy(self.labels[index])
        return img_tensor, label

    def __len__(self):
        return len(self.img_name_list)


def train_model(model, dataloaders, criterion_classifier,optimizer, num_epochs):
    since = time.time()

    val_mAP_history = []
    best_mAP = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # =========================================
        #  optimize the model on the training data
        # =========================================
        model.train()
        running_classifier_loss = 0
        batch = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(inputs,labels)
                loss_classifier = criterion_classifier(outputs, labels)
                # loss=0.90*loss_classifier+0.10*loss_decoder
                loss = loss_classifier
                loss.backward()
                optimizer.step()

            batch_classifier_loss = loss_classifier.item() * inputs.size(0)
            running_classifier_loss += batch_classifier_loss
            batch += 1

            print('Epoch:{}/{}, Batch:{}, Loss_classifier:{}'.format(epoch + 1, num_epochs, batch, loss_classifier.item()))

        epoch_classifier_loss = running_classifier_loss / len(dataloaders['train'].dataset)
        print('train classifier Loss: {:.4f}'.format(epoch_classifier_loss))
        with open('res50_pm_loss.txt', 'a') as f:
            f.write('Epoch:{}, Loss_classifier:{:.4f}\n'.format(epoch + 1, epoch_classifier_loss
                                                                                  ))

        # ===========================================
        #  evaluate the model on the validation data
        # ===========================================
        model.eval()
        running_classifier_loss = 0
        groud_truth = []
        pred_scores = []
        for inputs, labels in dataloaders['val']:
            inputs = inputs.cuda()
            labels = labels.cuda()

            with torch.set_grad_enabled(False):
                outputs  = model(inputs,labels)
                loss_classifier = criterion_classifier(outputs, labels)
                # loss = 0.90 * loss_classifier + 0.10 * loss_decoder
                loss =  loss_classifier

            running_classifier_loss = loss_classifier.item() * inputs.size(0)
            scores = torch.nn.functional.sigmoid(outputs)
            pred_scores.append(scores)
            groud_truth.append(labels)

        epoch_classifier_loss = running_classifier_loss / len(dataloaders['val'].dataset)
        pred_scores = torch.cat(tuple(pred_scores))
        groud_truth = torch.cat(tuple(groud_truth))
        mAP = average_precision_score(groud_truth, pred_scores)
        print('val classifier Loss: {:.4f},mAP: {:.4f}'.format(epoch_classifier_loss, mAP))
        with open('val-mAP-res50pm_repm.txt', 'a') as f:
            f.write('Epoch:{}, Loss:{}, mAP:{}\n'.format(epoch, loss, mAP))
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(),
                       'trained_models_cam-decoder/res50_nodrop_repm-%d.pth' % (epoch ))
        # torch.save(model.state_dict(),
        #            'trained_models_cam-decoder/res50_pm-%d.pth' % (epoch))
        val_mAP_history.append(mAP)
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val mAP: {:4f}'.format(best_mAP))

    return model



class DWConv(nn.Module):
    def __init__(self,in_channals, out_channels,kernel_size=3, stride=2, padding=1,dilation=1,bias=False):
        super(DWConv,self).__init__()
        self.con = nn.Sequential(nn.Conv2d(in_channals,in_channals,kernel_size,stride,padding,
                                           groups=in_channals,bias =bias),
                                 nn.Conv2d(in_channals,out_channels,1,1,0,dilation,
                                           groups=1,bias=bias),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.con(x)

        return  x



class GEmodule(nn.Module):
    def __init__(self,inchannal,outchannal):
        super(GEmodule,self).__init__()
        self.conv1 = DWConv(inchannal,outchannal)
        # self.dr1 = nn.Dropout()
        self.conv2 = DWConv(outchannal,outchannal)
        # self.dr2 = nn.Dropout()
        self.conv3 = DWConv(outchannal,outchannal)
        # self.dr3 = nn.Dropout()
        self.upsample = nn.Upsample(size=(28,28), mode='bilinear')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.dr1(out)
        out = self.conv2(out)
        # out = self.dr2(out)
        out = self.conv3(out)
        # out = self.dr3(out)
        out = self.upsample(out)
        out = self.sigmoid(out)
        out = torch.mul(residual,out)
        out += residual

        return out

class upsample(nn.Module):
    def __init__(self):
        super(upsample,self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
    def forward(self, x,low_map):
        x = self.upsample(x)
        x = x+low_map
        return  x

class UpsampleBlock(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size=3, stride=1, padding=1):
        super(UpsampleBlock,self).__init__()
        self.upsample =nn.Sequential(DWConv(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                                     nn.Upsample(scale_factor=2,mode='bilinear'))
    def forward(self, x):
        x = self.upsample(x)
        return  x

class merge(nn.Module):
     def __init__(self):
          super(merge,self).__init__()
          self.upsample1 = UpsampleBlock(2048,1024)
          self.upsample2 = UpsampleBlock(1024,512)
          self.upsample3 = UpsampleBlock(1024,512)
          self.GE = GEmodule(512, 512)
          # self.conv = DWConv(128,128,3,1,1)

     def forward(self, x,low_map,mid_map):
          x = self.upsample1(x)
          x = self.upsample2(x)
          mid_map = self.upsample3(mid_map)
          x = x+mid_map+low_map
          x = self.GE(x)
          # x = nn.functional.pixel_shuffle(x,2)
          # x = self.conv(x)
          return  x


class features (nn.Module):
    def __init__(self):
        super(features,self).__init__()
        self.resnet_layer = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        self.merge = merge()

    def forward(self, x):
        for name, module in self.resnet_layer._modules.items():
            x = module(x)
            if (name == '5'):
                low_map = x
            elif(name =="6"):
                mid_map = x
        x = self.merge(x, low_map,mid_map)
        return x

class classifer (nn.Module):
    def __init__(self):
        super(classifer,self).__init__()
        self.avg = nn.AvgPool2d(28)
        self.fc = nn.Linear(512, 20)

    def forward(self, x,label):
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features = features()
        self.classifier = classifer()

    def forward(self, x,label):
        x = self.features(x)
        x = self.classifier(x,label)
        return x



def get_cam_model(start_epoch):

    params_path = '/media/data/users/master/2018/zhanghan/trained_models_cam-decoder/res50_nodrop_repm-{}.pth'.format(start_epoch)
    model = Net()
    model.load_state_dict(torch.load(params_path))

    return model

if __name__=="__main__":
    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(model_ft.input_size),
            transforms.RandomCrop(input_size, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: VOCdataset(DATA_PATH.format(x), LABEL_PATH.format(x), data_transforms[x]) for x in
                      ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
    ['train', 'val']}


    net = Net()
    start_epoch = 7
    params_path = '/media/data/users/master/2018/zhanghan/trained_models_cam-decoder/res50_nodrop_repm-{}.pth'.format(start_epoch)
    net.load_state_dict(torch.load(params_path))

    print(net)

    net.cuda()

    params_to_update = net.parameters()
    optimizer = optim.Adam(params_to_update, lr=1e-5)
    criterion_classifier = nn.MultiLabelSoftMarginLoss()
    model = train_model(net, dataloaders_dict, criterion_classifier,optimizer, num_epochs)