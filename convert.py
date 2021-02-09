import numpy as np
from collections import defaultdict
import json
import copy

import os


def func(path, file_type):
    result = []
    for root, dirs, files in os.walk(path):
        print('print the absolute path of the directory...')
        for dir in dirs:
            print(os.path.join(root, dir))

        print('print the absolute path of the file...')
        for file in files:
            print(os.path.join(root, file))
            result.append(os.path.join(root, file))

        print('')
    return result


def convert2json(files, path):
    for file in files:
        file_name = file.split('/')[-1].replace('.txt', '.json')
        objects = []
        for line in open(file, "r", encoding='utf-8'):
            # 每个输入数据以逗号隔开
            items = line.strip("\n")
            dictinfo = json.loads(items)
            brand = dictinfo.get('brand', None)
            print(brand)
            dictinfo['brand'] = brand if brand is not None else brand
            objects.append(dictinfo)

        json_str = json.dumps(objects, indent=2, ensure_ascii=False)  # 注意这个indent参数
        with open(os.path.join(path + file_name), 'w', encoding='utf-8') as json_file:
            json_file.write(json_str)


def convert2bulkjson(files, path):
    for file in files:
        file_name = file.split('/')[-1].replace('.txt', '.json')
        with open(os.path.join(path + 'bulk'+file_name), 'w', encoding='UTF-8') as fin:
            for line in open(file, "r", encoding='utf-8'):

                # 每个输入数据以逗号隔开
                items = line.strip("\n")
                dictinfo = json.loads(items)
                # 添加index行
                new_data = {}
                new_data['index'] = {}
                new_data['index']['_index'] = "spu"
                new_data['index']['_id'] = dictinfo['spuId']
                temp = json.dumps(new_data).encode("utf-8").decode('unicode_escape')
                fin.write(temp)
                fin.write('\n')

                #原json对象处理为1行
                temp = json.dumps(dictinfo).encode("utf-8").decode('unicode_escape')
                fin.write(temp)
                fin.write('\n')



if __name__ == '__main__':
    files = func('/Users/gallup/Downloads/归档', '.txt')
    convert2bulkjson(files, './data/json/')
