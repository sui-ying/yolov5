from pathlib import Path
import os
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
from functools import reduce

origin_gt = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                    'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'sh',
                    'js', 'ha', 'ah', 'zj', 'bj', 'tj', 'cq', 'he', 'yn', 'ln', 'hl', 'hn', 'sd', 'sc', 'jx',
                    'hb', 'gs', 'sx', 'sn', 'jl', 'fj', 'gz', 'gd', 'gx', 'qh', 'hi', 'nx', 'xz', 'nm', 'xj',
                    'xue', 'gua', 'jiing', 'gang','inside', '-']

dst_gt = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",
                    "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
                    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "沪", "苏",
                    "豫", "皖", "浙", "京", "津", "渝", "冀", "滇", "辽", "黑", "湘", "鲁",
                    "川", "赣", "鄂", "甘", "晋", "陕", "吉", "闽", "黔", "粤", "桂", "青",
                    "琼", "宁", "藏", "蒙", "新", "学", "挂", "警", "港", "内", '-']
gt_map = {}

image_path = '/media/westwell/cv/zff/wellocean4_project/QingDao/scp/images/train/'
label_path = '/media/westwell/cv/zff/wellocean4_project/QingDao/scp/all_xml'
crnn_txt = '/media/westwell/cv/zff/wellocean4_project/QingDao/scp/plate_crnn.txt'
ignore_list = ['/media/westwell/cv/zff/wellocean4_project/QingDao/crnn_trucknum_sideplate/result_select_images2/2排',
               '/media/westwell/cv/zff/wellocean4_project/QingDao/crnn_trucknum_sideplate/result_select_images1/2排',
               '/media/westwell/cv/zff/wellocean4_project/QingDao/crnn_trucknum_sideplate/result_select_images1/1排'
               ]

# ignore_list = ['1ccf234f76c2180fc62f1353c325578_2021-05-25_0.jpg',
#                '1a4a139f3cea4271e67361119e45b63_2021-05-25_0.jpg',
#                '18981cde659b679cacc35e2de070961_2021-05-25_0.jpg',
#                '1182683dada2d936a2d358d3874cf69_2021-05-25_0.jpg',
#                '17779.bmp',
#                '17915.bmp',
#                '18310.bmp',
#                '21139.bmp',
#                '22019.bmp',
#                '22119.bmp',
#                '22461.bmp',
#                '24605.bmp',
#                '24610.bmp',
#                '24742.bmp',
#                '25925.bmp',
#                '25952.bmp',
#                '26306.bmp',
#                '26791.bmp',
#                '27049.bmp',
#                '28833.bmp',
#                '29001.bmp',
#                '29124.bmp',
#                '29125.bmp',
#                '29988.bmp',
#                '30349.bmp',
#                '30790.bmp',
#                '31344.bmp']

ignore_set = set()

for index in range(len(origin_gt)):
    gt_map[origin_gt[index]] = dst_gt[index]

for i in ignore_list:
    if Path(i).is_dir():
        iterations = Path(i).iterdir()
        for j in iterations:
            ignore_set.add(j.stem)
    else:
        ignore_set.add(Path(i).stem)

def get_label_dict(): # letter_mode can be 'upper', 'lower', None
    """
    get label dict
    :param mode: if this use train or test mode
    :param letter_mode: if letter be upper or lower or ignore
    :return: label dict
    """
    label_dict = {}
    invalid_label_list = []
    label_iterations = Path(label_path).iterdir()
    with tqdm(total=len(os.listdir(label_path))) as p_bar:
        p_bar.set_description('get label dict')
        for i in label_iterations:
            if i.stem in ignore_set:
                p_bar.update(1)
                continue
            name = i.stem
            tree = ET.parse(i.absolute())
            root = tree.getroot()
            valid = True
            li = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls in gt_map:
                    bndbox = obj.find('bndbox')
                    xmin = int(bndbox.find('xmin').text)
                    li.append((gt_map[cls],xmin))
                else:
                    if cls == '_':
                        cls = '-'
                        bndbox = obj.find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        li.append((gt_map[cls], xmin))
                    elif cls == 'jing':
                        cls = 'jiing'
                        bndbox = obj.find('bndbox')
                        xmin = int(bndbox.find('xmin').text)
                        li.append((gt_map[cls], xmin))
                    else:
                        print(f'---------{cls} is not in the gt class')
                        valid = False
                        break
            if valid:
                li = sorted(li, key=lambda item: item[1])
                # print('sorted list:',li)
                s = reduce(lambda a, b: a+b, map(lambda m:m[0], li))
                label_dict[name] = s
            else:
                invalid_label_list.append([name])
            p_bar.update(1)
    if invalid_label_list:
        print(invalid_label_list)
    return label_dict


def generate_datasets():
    """
    generate datasets
    :param mode:  if this use train or test mode
    :param letter_mode:  if letter be upper or lower or ignore
    :param check_vertical: check if the image is vertical
    :return:
    """
    # target_path = crnn_txt
    # w = open(target_path, 'w')
    # iterations = Path(image_path).iterdir()
    label_dict = get_label_dict()
    print('label dict length:', len(label_dict))
    # with tqdm(total=len(os.listdir(image_path))) as p_bar:
    #     p_bar.set_description('generate datasets')
    #     for i in iterations:
    #         if i.suffix == '.jpg' or i.suffix == '.png' or i.suffix == '.jpeg' or i.suffix == '.bmp':
    #             name = i.name[:-len(i.suffix)]
    #             if name in label_dict:
    #                 w.write(f'{i.name} {label_dict[name]}\n')
    #         p_bar.update(1)
    # w.close()

if __name__ == '__main__':
    generate_datasets()

