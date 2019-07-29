import os

voc2012_Main='G:/大二/实验室学习/voc2012/VOC2012/ImageSets/Main'
files = os.listdir(voc2012_Main)
result_List = set()
except_list=['train.txt','trainval.txt','val.txt']
for s in except_list:
    files.remove(s)

for file in files:
    if 'val' not in file:
        result_List.add(file)

s = {'table':1,'pig':2}
index =1
f= open('label_map.txt','w')
for file in result_List:
    f.write('\''+file[:-10]+'\': '+str(index)+',')
    index=index+1
