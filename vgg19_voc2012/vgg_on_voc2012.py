import argparse
import os

from utils import *

if __name__ == '__main__':
    # 处理命令行输入参数
    parser = argparse.ArgumentParser()
    parser.description = ('此程序用于训练vgg模型，训练过程将会在控制台输出，可实时监测，训练过程将记录在文件vgg_[vgg_type]_train_log.csv\n' +
                          '最后的epoch的训练结果将会保存为vgg_[vgg_type]_trained.pth\n' +
                          '训练过程中每个valid loss下降的epoch训练结果将会以state_dict的形式保存为文件vgg_[vgg_type]_epoch_[epoch_index]_dict.pth\n' +
                          '在服务器上训练时，请使用如下形式开始后台训练\n' +
                          '    >>nohup python vgg19_on_voc2012.py 必要的参数 > train_log.txt &\n' +
                          '查看训练进度时，请使用命令\n' +
                          '    >>cat train_log.txt\n' +
                          '需要停止训练时，使用ps+kill工具停止训练进程即可').replace('\n', os.linesep)
    parser.add_argument('dataset', help='voc2012数据库的根目录，此目录下必须包含VOCdevkit/VOC2012', type=str)
    parser.add_argument('batch_size', help='指定每个batch的数据量', type=int)
    parser.add_argument('epoch', help='指定epoch的大小', type=int)
    parser.add_argument('lr', help='指定学习率learning rate', type=float)
    parser.add_argument('num_works', help='指定数据集读取线程数', type=int)
    parser.add_argument('--use-cuda', help='是否使用GPU加速，需要加速时在命令行添加 --use-cuda', action='store_true', default=False)
    parser.add_argument('-vgg_type', help='指定vgg模型的版本，可选择：11、13、16、19', default=19, type=int)
    parser.add_argument('-prev_model', help='指定以前训练的vgg模型，模型必须与vgg type一致', default=None, type=str)

    # 初始化参数
    args = parser.parse_args()
    verbose = True
    shuffle = True
    voc2012_root = args.dataset
    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    num_works = args.num_works
    use_gpu = args.use_cuda
    vgg_type = args.vgg_type  # 11，13，16，19 中选择一个
    prev_model = None
    if args.prev_model:
        prev_model = args.prev_model
    log_save_file = 'vgg_{}_train_log.csv'.format(vgg_type)

    #     训练模型
    train_model(vgg_type, voc2012_root, batch_size, epoch, lr, log_save_file, use_gpu,
                shuffle, verbose, num_works, pre_train=True,
                prev_model=prev_model)  # 'vgg_{}_trained.pth'.format(vgg_type)
