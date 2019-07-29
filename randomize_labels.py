import os
import random

import sys


def parse_label_image(label_file_path,verbose=False):
    """
    解析label中的内容。经过遍历已知，样本类型只有0、1、-1三种

    :param label_file_path: label文件的位置
    :param verbose: verbose=True时在控制台打印文件解析的结果
    :return: label_dict:文件解析而成的dict,key代表图片编号，value代表样本类型;  value_dict:key表示样本类型，value表示此类型样本所有图片的编号
    """
    label_strings=open(label_file_path).readlines()
    label_dict = {}
    value_dict={}
    value_set=set()
    for label in label_strings:
        s=label.split()
        assert len(s)==2
        label_dict[s[0]]=int(s[1])
        value_set.add(int(s[1]))

    if verbose:
        print(label_file_path.split('/')[-1]+'：样本取值类型：' + str(value_set))
    show_sample_detail(label_dict, value_dict)
    return label_dict,value_dict


def show_sample_detail(label_dict, value_dict=None,verbose=False):
    """
    输出样本的具体细节到控制台
    """
    if value_dict is None:
        value_dict = {}
    value_set=set()
    for k in label_dict:
        value_set.add(label_dict[k])
    for i in value_set:
        value_dict[i] = set()
    for k in label_dict.keys():
        value_dict[label_dict[k]].add(k)
    if verbose:
       for i in list(value_set):
           print('  ' + str(i) + '样本数量：' + str(len(value_dict[i])))

def get_files_absolute_path(dir_path):
    '''
    获取训练集中ImageSets/Main文件夹中所有的训练集合。并且去除3个非训练集文件 'train.txt','trainval.txt','val.txt'
    :param dir_path: ImageSets/Main文件夹的绝对路径
    :return: 文件夹中所有的训练集、验证集文件
    '''
    file_list=os.listdir(dir_path)
    except_list=['train.txt','trainval.txt','val.txt']
    for s in except_list:
        if s in file_list:
            file_list.remove(s)
    if dir_path[-1] !='/':
        dir_path=dir_path+'/'
    for i in range(len(file_list)):
        file_list[i]=dir_path+ file_list[i]
    return file_list


def randomize_label(label_file_path, image_data_path, out_put_dir='noise_samples', ratio=0.05,verbose=False):
    """
    随机向数据集中加入噪声\n
    加入噪声的方法：将数量为x的负样本改变为正样本，记s为原本正样本的数量。取x使得ratio=x/s （忽略舍入带来的误差）
    输出文件后面带有的小数代表噪声比例

    :param out_put_dir: 输出文件的路径
    :param ratio:加入噪声的比例
    :param label_file_path:数据集文件的路径
    :param image_data_path:图片数据的路径，用于验证数据集的合法性
    :param verbose: verbose==true时显示操作具体细节
    """
    ratio = float('%0.2f'%ratio)
    if verbose:
        print('文件:{} 噪声:{}'.format(str(label_file_path.split('/')[-1]),ratio).center(100,'-'))


# 获取图片集合中所有图片的名称,并且检验其是否在总图片集合中
    image_list=os.listdir(image_data_path)
    for i in range(len(image_list)):
        image_list[i]=image_list[i][:11]
    image_set = set(image_list)
    label_image_dict,value_dict=parse_label_image(label_file_path,verbose=verbose)
    for key in label_image_dict.keys():
        if key not in image_set:
            print("文件不存在，数据集中不办包含："+key)

    # 创建文件夹
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    change_number=int((ratio * len(value_dict[1]))/(1-ratio))
    change_keys=random.sample(value_dict[-1],change_number)
    for k in change_keys:
        label_image_dict[k]=1
    if verbose:
        print('修改的样本类型的图片：'+ str(change_keys))

    if out_put_dir[-1]!='/':
        out_put_dir = out_put_dir +'/'
    output_file_name=out_put_dir+label_file_path.split('/')[-1].split('.')[0]+'_'+str(ratio)+'.txt'
    output_file=open(output_file_name,'w')
    for k in label_image_dict:
        output_file.write(k+' '+str(label_image_dict[k])+'\n')
    if verbose:
        print('修改后数据集(噪声比例:'+str(ratio) +'):')
        show_sample_detail(label_image_dict)


if __name__ == '__main__':
    '''
    请输入voc2012的根目录参数作为参数
    '''
    if sys.argv.__len__() !=2:
        print('请输入voc2012的根目录参数作为参数，且只输入一个参数')
        exit(-1)

    voc2012=sys.argv[1]
    if voc2012[-1]!='/':
        voc2012=voc2012+'/'
    image_data_path=voc2012+'JPEGImages'
    label_file_dir=voc2012+'ImageSets/Main'
    files= get_files_absolute_path(label_file_dir)
    ratio = 0.1
    print('生成噪声比例：'+str(ratio))
    for file in files:
        randomize_label(file, image_data_path, ratio=ratio, verbose=True)
