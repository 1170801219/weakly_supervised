import os


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