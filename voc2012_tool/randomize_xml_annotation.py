import os
import random
import shutil
import xml.dom.minidom

from torchvision.datasets import VOCDetection

backup_dir = 'backup_dir'


def get_used_annotations(root):
    train = VOCDetection(root=root, year='2012', image_set='train')
    trainval = VOCDetection(root=root, year='2012', image_set='trainval')
    val = VOCDetection(root=root, year='2012', image_set='val')
    train_images = train.images
    trainval_images = trainval.images
    val_images = val.images

    # 测试是否所有图片都被使用到
    used_annotations = [j[-15:-4] for j in train_images]
    for i in [j[-15:-4] for j in train_images]:
        used_annotations.append(i)
    for i in [j[-15:-4] for j in trainval_images]:
        used_annotations.append(i)
    for i in [j[-15:-4] for j in val_images]:
        used_annotations.append(i)
    used_annotations_set = set(used_annotations)
    return used_annotations_set


def copy_files(src_dir, dst_dir):
    '''
    将一个文件夹里的所有文件移动到另外一个文件夹中
    '''
    files = [i for i in os.listdir(src_dir)]
    for f in files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        print('copy {} --> {}'.format(os.path.join(src_dir, f), os.path.join(dst_dir, f)))


def backup_annotations(dataset_path):
    annotation_dir = os.path.join(dataset_path, r'VOCdevkit/VOC2012/Annotations')
    try:
        os.makedirs(backup_dir)
    except:
        pass
    #     备份以前的annotation数据
    copy_files(annotation_dir, backup_dir)


def restore_annotations(dataset_path):
    annotation_dir = os.path.join(dataset_path, r'VOCdevkit/VOC2012/Annotations')
    #     还原以前的数据
    copy_files(backup_dir, annotation_dir)


def randomize_xml(dataset_path, ratio, correct_annotation_dir):
    labels = ['bird', 'train', 'tvmonitor', 'pottedplant', 'cat', 'sofa', 'car', 'aeroplane', 'dog', 'cow',
              'diningtable', 'person', 'horse', 'motorbike', 'boat', 'sheep', 'bottle', 'bus', 'bicycle', 'chair']
    used_annotations = get_used_annotations(dataset_path)
    number_to_change = int(len(used_annotations) * ratio)
    files_to_change = random.sample(used_annotations, number_to_change)
    annotation_dir = os.path.join(dataset_path, r'VOCdevkit/VOC2012/Annotations')
    for file in files_to_change:
        change_xml(os.path.join(correct_annotation_dir, file + '.xml'),
                   os.path.join(annotation_dir, file + '.xml'),
                   labels)


def change_xml(src_xml_file, dst_xml_file, labels):
    dom = xml.dom.minidom.parse(src_xml_file)
    root = dom.documentElement
    objects = root.getElementsByTagName('object')
    for o in objects:
        prev_tag = o.getElementsByTagName('name')[0].firstChild.data
        post_tag = random.choice(labels)
        while post_tag == prev_tag:
            post_tag = random.choice(labels)
        o.getElementsByTagName('name')[0].firstChild.data = post_tag

    with open(dst_xml_file, 'w') as f:
        dom.writexml(f)


if __name__ == '__main__':
    data_path = r'G:\大二\实验室学习'
    # backup_annotations(data_path)
    # randomize_xml(data_path,0.05,backup_dir)
    restore_annotations(data_path)
