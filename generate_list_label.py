from experimental_code.randomize_labels import get_files_absolute_path, parse_label_image

label_map ={'bird': 1,'dog': 2,'sofa': 3,'cow': 4,'tvmonitor': 5,'person': 6,'bicycle': 7,'motorbike': 8,'diningtable': 9,'bottle': 10,'chair': 11,'boat': 12,'car': 13,'cat': 14,'sheep': 15,'train': 16,'pottedplant': 17,'aeroplane': 18,'horse': 19,'bus': 20}

def generate_list_label(path,verbose=False):
    '''
    生成用于产生lmdb的label文件
    :param path: 数据集的路径，其中文件的格式与VOC2012/ImageSet/Main下的文件相同
    :param verbose: 显示详细描述信息
    '''
    file_list = get_files_absolute_path(path)
    train_files =set()
    trainval_files = set()
    val_files=set()
    for file in file_list:
        if file.split('/')[-1].split('_')[1][:5] == 'train':
            train_files.add(file)
            continue
        elif file.split('/')[-1].split('_')[1][:8] == 'trainval':
            trainval_files.add(file)
            continue
        elif file.split('/')[-1].split('_')[1][:3] == 'val':
            val_files.add(file)
            continue
        assert False

    train_file =open('train.txt','w')
    trainval_file =open('trainval.txt','w')
    val_file =open('val.txt','w')
    save_to_file(train_file, train_files, verbose)
    save_to_file(trainval_file, trainval_files, verbose)
    save_to_file(val_file, val_files, verbose)

def save_to_file(file, files_to_read, verbose):
    '''
    将内容保存至文件
    '''
    index = 0
    for f in files_to_read:
        label = f.split('/')[-1].split('_')[0]
        label_image_dict, value_dict = parse_label_image(f, verbose=verbose)
        index = index + 1
        # 将标签内容写入文件中
        for image in value_dict[1]:
            file.write(image + '.jpg ' + str(label_map[label]) + '\n')
    if verbose:
        print("共有" + str(index) + '个类别写入数据集中')


if __name__ =='__main__':
    Main_path = 'G:/大二/实验室学习/voc2012/VOC2012/ImageSets/Main'
    generate_list_label(Main_path,verbose=False)