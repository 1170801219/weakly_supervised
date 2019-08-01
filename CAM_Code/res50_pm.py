import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms

# from sklearn.metrics import average_precision_score
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs=30
batch_size=62
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


def train_model(model, dataloaders, criterion_classifier,criterion_decoder,optimizer, num_epochs):
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
        running_decoder_loss=0
        batch = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs,img = model(inputs,labels)
                loss_classifier = criterion_classifier(outputs, labels)
                loss_decoder=criterion_decoder(img,inputs)
                # loss=0.90*loss_classifier+0.10*loss_decoder
                loss = 1* loss_classifier + 0.1 * loss_decoder
                loss.backward()
                optimizer.step()

            batch_classifier_loss = loss_classifier.item() * inputs.size(0)
            batch_decoder_loss=loss_decoder.item() * inputs.size(0)
            running_classifier_loss += batch_classifier_loss
            running_decoder_loss += batch_decoder_loss
            batch += 1

            print('Epoch:{}/{}, Batch:{}, Loss_classifier:{},Loss_decoder:{}'.format(epoch + 1, num_epochs, batch, loss_classifier.item(),loss_decoder.item()))

        epoch_classifier_loss = running_classifier_loss / len(dataloaders['train'].dataset)
        epoch_decoder_loss=running_decoder_loss / len(dataloaders['train'].dataset)
        print('train classifier Loss: {:.4f}'.format(epoch_classifier_loss))
        print('train decoder Loss: {:.4f}'.format(epoch_decoder_loss))
        with open('res50_pm_loss.txt', 'a') as f:
            f.write('Epoch:{}, Loss_classifier:{:.4f},Loss_decoder:{:.4f}\n'.format(epoch + 1, epoch_classifier_loss,
                                                                                    epoch_decoder_loss))

        # ===========================================
        #  evaluate the model on the validation data
        # ===========================================
        model.eval()
        running_classifier_loss = 0
        running_decoder_loss = 0
        groud_truth = []
        pred_scores = []
        for inputs, labels in dataloaders['val']:
            inputs = inputs.cuda()
            labels = labels.cuda()

            with torch.set_grad_enabled(False):
                outputs, img = model(inputs,labels)
                loss_classifier = criterion_classifier(outputs, labels)
                loss_decoder = criterion_decoder(img, inputs)
                # loss = 0.90 * loss_classifier + 0.10 * loss_decoder
                loss = 1 * loss_classifier + 0.1 * loss_decoder

            running_classifier_loss = loss_classifier.item() * inputs.size(0)
            running_decoder_loss = loss_decoder.item() * inputs.size(0)
            scores = torch.nn.functional.sigmoid(outputs)
            pred_scores.append(scores)
            groud_truth.append(labels)

        epoch_classifier_loss = running_classifier_loss / len(dataloaders['val'].dataset)
        epoch_decoder_loss=running_decoder_loss / len(dataloaders['val'].dataset)
        pred_scores = torch.cat(tuple(pred_scores))
        groud_truth = torch.cat(tuple(groud_truth))
        mAP = average_precision_score(groud_truth, pred_scores)
        print('val classifier Loss: {:.4f},val decoder Loss: {:.4f},mAP: {:.4f}'.format(epoch_classifier_loss,
                                                                                         epoch_decoder_loss, mAP))
        with open('val-mAP-res50pm.txt', 'a') as f:
            f.write('Epoch:{}, Loss:{}, mAP:{}\n'.format(epoch, loss, mAP))
        if mAP > best_mAP:
            best_mAP = mAP
            torch.save(model.state_dict(),
                       'trained_models_cam-decoder/res50_de_pm-%d.pth' % (epoch ))
        # torch.save(model.state_dict(),
        #            'trained_models_cam-decoder/res50_pm-%d.pth' % (epoch))
        val_mAP_history.append(mAP)
        print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val mAP: {:4f}'.format(best_mAP))

    return model

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor,kernel_size=3, stride=1, padding=1):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(0.2,inplace=True),nn.Upsample(scale_factor=scale_factor,mode='bilinear'))

    def forward(self, x):
        x=self.deconv(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv1 = DeconvBlock(512,256,2)
        self.deconv2 = DeconvBlock(256,128,2)
        self.deconv3 = DeconvBlock(128, 3, 2)


    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)



        return x

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
        self.dr1 = nn.Dropout()
        self.conv2 = DWConv(outchannal,outchannal)
        self.dr2 = nn.Dropout()
        self.conv3 = DWConv(outchannal,outchannal)
        self.dr3 = nn.Dropout()
        self.upsample = nn.Upsample(size=(28,28), mode='bilinear')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.dr1(out)
        out = self.conv2(out)
        out = self.dr2(out)
        out = self.conv3(out)
        out = self.dr3(out)
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

     def forward(self, x,low_map,mid_map):
          x = self.upsample1(x)
          x = self.upsample2(x)
          mid_map = self.upsample3(mid_map)
          x = x+mid_map+low_map
          x = self.GE(x)
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
        map = x
        return x,map

class classifer (nn.Module):
    def __init__(self):
        super(classifer,self).__init__()
        self.avg = nn.AvgPool2d(28)
        self.fc = nn.Linear(512, 20)

    def forward(self, x,map,label):
        reweight = torch.zeros(label.size(0),512,1,1)
        reweight = reweight.cuda()
        x = self.avg(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        weight = self.fc.weight
        weight = weight.data.cpu().numpy()
        weight = Variable(torch.from_numpy(weight))
        weight = weight.cuda()
        for i in range(0, reweight.size(0)):
            t = torch.nonzero(label[i])
            for j in range(0, t.size(0)):
                q = weight[t[j]].view(512, 1, 1)
                reweight[i] += q
        map = map * reweight
        return x,map

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features = features()
        self.classifier = classifer()
        self.decoder = Decoder()

    def forward(self, x,label):
        x, map = self.features(x)
        x, map = self.classifier(x,map,label)
        pic = self.decoder(map)
        return x,pic




def get_cam_model(start_epoch):

    params_path = '/media/data/users/master/2018/zhanghan/trained_models_cam-decoder/res50_de_pm-{}.pth'.format(start_epoch)
    model = Net()
    model.load_state_dict(torch.load(params_path))
    weight = model.classifier.fc.weight
    print(weight.shape)
    weight = weight.data.numpy()
    np.savetxt("weight", weight)
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
    start_epoch = 3
    # params_path = '/media/data/users/master/2018/zhanghan/trained_models_cam-decoder/res50_de_pm-{}.pth'.format(start_epoch)
    # net.load_state_dict(torch.load(params_path))

    print(net)

    net.cuda()

    params_to_update = net.parameters()
    optimizer = optim.Adam(params_to_update, lr=1e-5)
    criterion_classifier = nn.MultiLabelSoftMarginLoss()
    criterion_decoder=nn.SmoothL1Loss()
    model = train_model(net, dataloaders_dict, criterion_classifier,criterion_decoder, optimizer, num_epochs)