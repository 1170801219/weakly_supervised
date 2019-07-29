import os


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
