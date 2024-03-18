import numpy as np
import json

# 直接设置label中有一个相似的即可完成匹配
label_data_train = json.load(open('/home/whc/Downloads/annotations_trainval2014/annotations/instances_train2014.json', 'r'))
label_data_val = json.load(open('/home/whc/Downloads/annotations_trainval2014/annotations/instances_val2014.json', 'r'))

label_data = label_data_train['annotations'] + label_data_val['annotations']

id_category = {}

for tmp in label_data:
    try:
        if len(id_category[tmp['image_id']]) == 0:
            id_category[tmp['image_id']] = []
    except:
        id_category[tmp['image_id']] = []
        
    id_category[tmp['image_id']].append(tmp['category_id'])

for key, value in id_category.items():
    id_category[key] = list(set(value))


# 设置阈值为1，只要有一个label是相似的，就统计出来
from tqdm import tqdm
import random

asym = {}
asym_index = {}
for key, value in tqdm(id_category.items()):
    
    try:
        if len(asym[key]) == 0:
            asym[key] = []
    except:
        asym[key] = []
        
    for tmp_key, tmp_value in id_category.items():
        if key != tmp_key:
            for tmp in value:
                if tmp in tmp_value:
                    asym[key].append(tmp_key)
                    
    asym[key] = list(set(asym[key]))
    rand_index = random.randint(0, len(asym[key]))
    asym_index[key] = asym[key][rand_index-1]

fr = open('asym.json', 'w')
fl = open('asym_index.json', 'w')
json.dump(asym, fr)
json.dump(asym_index, fl)
fr.close()
fl.close()