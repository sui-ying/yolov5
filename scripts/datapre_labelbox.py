#encoding=utf-8
import json
import xml.etree.ElementTree as ET
import cv2
import os
import shutil
import random
import argparse
import sys

sys.path.append("/cv/xyc/myscripts")
from my_io import JsonHandler
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance
from convert import xxyy_convert_xywh

from convert import xxyy_convert_xywh, xywh_convert_xxyy

import sys
sys.path.append("../../../myscripts")
from figure import draw_bar
from scripts_for_image import diff_img_lablefile_and_move_extra, rgb2gray_for_imgset_and_save
from tools import not_exists_path_make_dir, not_exists_path_make_dirs
from file_operate import Count_the_number_of_directory
from augment import contrastEnhancement, brightnessEnhancement, colorEnhancement, sharp_enhance, noise_enhance

def convert_annotation(image_id, workpath):
    """
    read xml INFO, write 'cls, x, y, w, h' to lables
    """
    in_file = open(os.path.join(workpath, 'xml', image_id + '.xml'))
    out_file = open(os.path.join(workpath, 'info', image_id + '.txt'), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = xxyy_convert_xywh((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def add_additional_dataset_to_train(wdir):
    sets = ['train']

    imgpath = os.path.join(wdir, 'img')  # img path
    xmlpath = os.path.join(wdir, 'xml')  # xml path

    # output
    labelpath = os.path.join(wdir, 'info')  # xml 2 info: [cls, x, y, w, h]
    if not os.path.exists(labelpath):
        os.makedirs(labelpath)

    for image_set in sets:
        image_ids = open(os.path.join(wdir, 'add_' + image_set + '.txt')).read().strip().split()

        # 1. images, labels
        image_list_file = open(os.path.join(wdir, 'images_' + image_set + '.txt'), 'a+')
        labels_list_file = open(os.path.join(wdir, 'labels_' + image_set + '.txt'), 'a+')
        with tqdm(total=len(image_ids)) as p_bar:
            for image_id in image_ids:
                image_list_file.write('%s.jpg\n' % (image_id))
                labels_list_file.write('%s.txt\n' % (image_id))
                if image_id + '.xml' in os.listdir(os.path.join(wdir, 'xml')):
                    convert_annotation(image_id, wdir)  # 如果标签已经是txt格式，将此行注释掉，所有的txt存放到all_labels文件夹。
                else:
                    continue
                p_bar.update(1)

        image_list_file.close()
        labels_list_file.close()

    # output path for train, val
    img_train_path = os.path.join(wdir, 'images/train')
    label_train_path = os.path.join(wdir, 'labels/train')

    # 按照划分好的训练文件的路径搜索目标，并将其复制到yolo格式下的新路径
    # train
    copy_file(img_train_path, os.path.join(wdir, 'images_train.txt'), imgpath)
    copy_file(label_train_path, os.path.join(wdir, 'labels_train.txt'), labelpath)

    # extend add_train.txt to imagesets/train.txt
    # open(os.path.join(wdir, 'add_train.txt')).read().strip().split()


def labeled_visualization(wdir, colors, flag_crop_str=False):
    # classes = ['hole', 'silver']
    # sets = ['train', 'val', 'test']

    imgs_path = os.path.join(wdir, 'img')  # img path
    info_path = os.path.join(wdir, 'info')  # labels path

    # Output
    retangele_img_path = os.path.join(wdir, 'all_visualization')  # 保存画框后图片的路径
    if not os.path.exists(retangele_img_path):
        os.makedirs(retangele_img_path)

    if not flag_crop_str:
        with tqdm(total=len(os.listdir(info_path))) as p_bar:
            p_bar.set_description('visual for labeled data')
            for file in os.listdir(info_path):

                if len(file.split('.')) > 2:
                    file_name = file[:-4]  # for file name have more than 1 dot
                else:
                    file_name = file.split('.')[0]

                if os.path.exists(os.path.join(imgs_path, file_name + '.png')):
                    img_name = os.path.join(imgs_path, file_name + '.png')
                elif os.path.exists(os.path.join(imgs_path, file_name + '.jpg')):
                    img_name = os.path.join(imgs_path, file_name + '.jpg')
                else:
                    print(file_name)
                    continue

                img = cv2.imread(img_name, -1)
                height, width = img.shape[0], img.shape[1]
                labelled = img

                with open(os.path.join(info_path, file), 'r') as label_info:
                    lines = label_info.readlines()
                    for i in range(len(lines)):
                        cls, x, y, w, h = lines[i].split(' ')

                        xmin, ymin, xmax, ymax = xywh_convert_xxyy((width, height), [float(x), float(y), float(w), float(h)])
                        # xmin = float(x) * float(width)
                        # ymin = float(y) * float(height)
                        # xmax = xmin + float(w) * float(width)
                        # ymax = ymin + float(h) * float(height)
                        # print(xmin, ymin, xmax, ymax)
                        labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[int(cls)], 1)
                        labelled = cv2.putText(labelled, classes[int(cls)], (int(xmin), int(ymin)-2), cv2.FONT_HERSHEY_PLAIN, 2,
                                                           colors[int(cls)], 1)  # fontscale, thickness
                cv2.imwrite(os.path.join(retangele_img_path, file_name + '.jpg'), labelled)

                p_bar.update(1)
    else:
        print("crop letter/digit and save...")
        for i in range(len(classes)):
            if not os.path.exists(os.path.join(wdir, 'crop_str', classes[i])):
                os.makedirs(os.path.join(wdir, 'crop_str', classes[i]))

        with tqdm(total=len(os.listdir(info_path))) as p_bar:
            p_bar.set_description('visual for labeled data')
            for file in os.listdir(info_path):

                if len(file.split('.')) > 2:
                    file_name = file[:-4]  # for file name have more than 1 dot
                else:
                    file_name = file.split('.')[0]

                if os.path.exists(os.path.join(imgs_path, file_name + '.png')):
                    img_name = os.path.join(imgs_path, file_name + '.png')
                elif os.path.exists(os.path.join(imgs_path, file_name + '.jpg')):
                    img_name = os.path.join(imgs_path, file_name + '.jpg')
                else:
                    print("Not exit img: ", file_name)
                    continue

                img = cv2.imread(img_name, -1)
                height, width = img.shape[0], img.shape[1]
                labelled = img

                with open(os.path.join(info_path, file), 'r') as label_info:
                    lines = label_info.readlines()
                    for i in range(len(lines)):
                        cls, x, y, w, h = lines[i].split(' ')

                        xmin, ymin, xmax, ymax = xywh_convert_xxyy((width, height),
                                                                   [float(x), float(y), float(w), float(h)])
                        # xmin = float(x) * float(width)
                        # ymin = float(y) * float(height)
                        # xmax = xmin + float(w) * float(width)
                        # ymax = ymin + float(h) * float(height)
                        # print(xmin, ymin, xmax, ymax)

                        labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                 colors[int(cls)], 2)
                        labelled = cv2.putText(labelled, classes[int(cls)], (int(xmin), int(ymin) - 2),
                                               cv2.FONT_HERSHEY_PLAIN, 2,
                                               colors[int(cls)], 2)

                        try:
                            _cropImg = img[ymin: ymax, xmin: xmax]  # crop ymin: ymax, xmin: xmax
                            _new_jpg_file = os.path.join(wdir, "crop_str", classes[int(cls)], file_name + str(i) + ".jpg")  # saved path
                            cv2.imwrite(_new_jpg_file, _cropImg)
                        except:
                            continue
                cv2.imwrite(os.path.join(retangele_img_path, file_name + '.jpg'), labelled)

                p_bar.update(1)


def train_val_test_visualization(wdir, colors):
    # classes = ['hole', 'silver']
    # sets = ['train', 'val', 'test']

    for set in sets:
        # Input
        imgs_path = os.path.join(wdir, 'images/%s' % (set))  # img path
        info_path = os.path.join(wdir, 'labels/%s' % (set))  # labels path

        # Output
        retangele_img_path = os.path.join(wdir, 'visualization/%s' % (set))  # 保存画框后图片的路径
        if not os.path.exists(retangele_img_path):
            os.makedirs(retangele_img_path)

        with tqdm(total=len(os.listdir(info_path))) as p_bar:
            p_bar.set_description('visual for train val test data')
            for file in os.listdir(info_path):

                if len(file.split('.')) > 2:
                    file_name = file[:-4]  # for file name have more than 1 dot
                else:
                    file_name = file.split('.')[0]

                if os.path.exists(os.path.join(imgs_path, file_name + '.png')):
                    img_name = os.path.join(imgs_path, file_name + '.png')
                elif os.path.exists(os.path.join(imgs_path, file_name + '.jpg')):
                    img_name = os.path.join(imgs_path, file_name + '.jpg')
                else:
                    print(file_name)
                    continue

                img = cv2.imread(img_name, -1)
                height, width = img.shape[0], img.shape[1]
                labelled = img

                with open(os.path.join(info_path, file), 'r') as label_info:
                    lines = label_info.readlines()
                    for i in range(len(lines)):
                        cls, x, y, w, h = lines[i].split(' ')

                        xmin, ymin, xmax, ymax = xywh_convert_xxyy((width, height), [float(x), float(y), float(w), float(h)])
                        # xmin = float(x) * float(width)
                        # ymin = float(y) * float(height)
                        # xmax = xmin + float(w) * float(width)
                        # ymax = ymin + float(h) * float(height)

                        labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[int(cls)], 2)
                        labelled = cv2.putText(labelled, classes[int(cls)], (int(xmin), int(ymin)-2), cv2.FONT_HERSHEY_PLAIN, 2,
                                                           colors[int(cls)], 2)
                cv2.imwrite(os.path.join(retangele_img_path, file_name + '.jpg'), labelled)

                p_bar.update(1)


def bbox_resize_nopad(iw, ih, box, nw, nh):
    """
    https://blog.csdn.net/qq_28949847/article/details/106154492
    :param img: 图片名
    :param box: 原box
    :param nw: 改变后的宽度
    :param nh: 改变后的高度
    :return:
    """
    # 将box进行调整
    box_resize = []
    for boxx in box:
        boxx[1] = str(int(int(boxx[1]) * (nw / iw)))
        boxx[2] = str(int(int(boxx[2]) * (nh / ih)))
        boxx[3] = str(int(int(boxx[3]) * (nw / iw)))
        boxx[4] = str(int(int(boxx[4]) * (nh / ih)))
        box_resize.append(boxx)
    return box_resize


def xml2info(imgpath, xmlpath, mode_type=0, visual=False, box_offset=False, resize_size=None):
    """
    :param imgpath: img path
    :param xmlpath: xml path
    :param mode_type: 0 for usual data, 1 for plate
    :param visual: visualization for image
    :param box_offset: flag for Calculate the bbox offset
    :param resize_size: (width, height) of resize size
    :return:
    """
    with tqdm(total=len(os.listdir(imgpath))) as p_bar:
        p_bar.set_description('xml2info')
        for file in os.listdir(imgpath):
            bbox_xywh= []
            img = cv2.imread(os.path.join(imgpath, file))
            height, width, channel = img.shape
            name, suffix = os.path.splitext(file)
            # 1. read xml to info
            if os.path.exists(os.path.join(xmlpath, name + ".xml")):
                # lst_xyxy, parse_info = read_xml_to_lst(xmlpath, name, mode_type, box_offset=box_offset, resize_size=resize_size)
                bbox_xyxy = read_xml_to_lst(xmlpath, name, mode_type)
            else:
                shutil.move(os.path.join(imgpath, file), error_img_path)
                print(name + ".xml not exist")
                p_bar.update(1)
                continue
            if len(bbox_xyxy) == 0:
                print("The len(info)=0: ", file)
                shutil.move(os.path.join(imgpath, file), os.path.join(wdir, "error/img"))
                shutil.move(os.path.join(xmlpath, name + ".xml"), os.path.join(wdir, "error/xml"))
                p_bar.update(1)
                continue

            # 2. img and box resize
            if box_offset:
                # 对图像进行缩放并且进行长和宽的扭曲
                img = cv2.resize(img, (resize_size[0], resize_size[1]), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(new_img_path, file), img)
                # INTER_NEAREST - 最邻近插值
                # INTER_LINEAR - 双线性插值，如果最后一个参数你不指定，默认使用这种方法
                # INTER_CUBIC - 4x4像素邻域内的双立方插值
                # INTER_LANCZOS4 - 8x8像素邻域内的Lanczos插值
                bbox_xyxy = bbox_resize_nopad(width, height, bbox_xyxy, resize_size[0], resize_size[1])  # np padding

            # 3. write label info to text
            for i in range(len(bbox_xyxy)):
                cls_index, xmin, ymin, xmax, ymax = bbox_xyxy[i]
                newx, newy, neww, newh = xxyy_convert_xywh((resize_size[0], resize_size[1]),
                                                           [float(xmin), float(ymin), float(xmax), float(ymax)])
                bbox_xywh.append([cls_index, newx, newy, neww, newh])
            out_file = open(os.path.join(infopath, name + '.txt'), 'w')
            for i in range(len(bbox_xywh)):
                line = str(bbox_xywh[i][0]) + " " \
                       + str(bbox_xywh[i][1]) + " " \
                       + str(bbox_xywh[i][2]) + " " \
                       + str(bbox_xywh[i][3]) + " " \
                       + str(bbox_xywh[i][4]) + "\n"
                out_file.write(line)
            out_file.close()

            # 3. visual
            if visual:
                labelled = img
                for i in range(len(bbox_xyxy)):
                    cls, xmin, ymin, xmax, ymax = bbox_xyxy[i]
                    labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[int(cls)], 1)
                    labelled = cv2.putText(labelled, classes[int(cls)], (int(xmin), int(ymin) - 2), cv2.FONT_HERSHEY_PLAIN,
                                           2, colors[int(cls)], 1)   # font scale, thickness
                cv2.imwrite(os.path.join(all_visual_path, name + '.jpg'), labelled)
            p_bar.update(1)


def read_xml_to_lst(xmlpath, image_id, plate_mode_type=0):
    """
    :param xmlpath: xml path
    :param image_id:
    :param plate_mode_type: default=0, 1 for plate
    :return: a 2d lst [[cls, xmin, ymin, xmax, ymax], [...]]
    """

    bbox_xyxy = []  # xyxy
    in_file = open(os.path.join(xmlpath, '%s.xml' % (image_id)))
    tree = ET.parse(in_file)
    root = tree.getroot()
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)

    for obj in root.iter('object'):
        if plate_mode_type == 0:  # for all data, except plate
            cls = obj.find('name').text
            cls = cls.replace('/', '_')
            if cls not in classes:
                continue
            # if cls == "ao":
            #     break
            # if len(cls) == 1:
            #     cls = cls.upper()
            # lorry
            # if cls == "tray-vertical" or cls == "tray":
            #     cls = "tray"
            # elif cls == "container-vertical" or cls == "container":
            #     cls = "container"
            # elif cls == "truckhead-down" or cls == "truckfront":
            #     cls = "truckfront"
            # elif cls == 'lorry-down' or cls == 'lorry-up' or cls == 'lorry':
            #     cls = "lorry"
            # elif cls == "truckhead-up" or cls == "truckback":
            #     cls = "truckback"
            # else:
            #     continue
            # print(cls)
        else:   # for plate
            # todo: simplify plate code
            tmp_name = obj.find('name').text
            plate_type = tmp_name.split("/")[-1]
            # modify here
            if "/" in tmp_name:
                # sanzahuo
                if plate_type == "WHITE":
                    cls = "white_plate"
                elif plate_type == "orange_plate":
                    cls = "yellow_plate"
                elif plate_type == "NE":
                    cls = "NE_plate"
                else:
                    cls = plate_type
                # if plate_type == "WHITE" or plate_type == "GREEN" or plate_type == "YELLOW":
                # cls = "plate"
        xmlbox = obj.find('bndbox')
        xmin, xmax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text)
        ymin, ymax = int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
        bbox_xyxy.append([classes.index(cls), xmin, ymin, xmax, ymax])

    return bbox_xyxy


def count_class_frequency(enhance_path, infopath, cls_nums, classes=None, flag_plot=True):
    for i in range(len(classes)):
        if not os.path.exists(os.path.join(enhance_path, classes[i])):
            os.makedirs(os.path.join(enhance_path, classes[i]))

    for file in os.listdir(infopath):
        f_info = open(os.path.join(infopath, file), 'r').readlines()
        try:
            for i in range(len(f_info)):
                cls = classes[int(f_info[i].split(" ")[0])]
                shutil.copy(os.path.join(infopath, file), os.path.join(enhance_path, cls))
                cls_nums.update([(cls, cls_nums[cls] + 1)])  # count subclass frequency
        except:
            print(file)

    print(cls_nums)
    class_frequency = open(os.path.join(enhance_path, 'class_frequency.txt'), 'w')
    class_frequency.write(json.dumps(cls_nums, indent=2))
    class_frequency.write('\n')
    class_frequency.close()

    if flag_plot:
        draw_bar(cls_nums, enhance_path)


def enhance_img_xml(wdir, imgpath, avg_nums, box_offset=False):
    """
    Aug: Brightness, contrast
    avg_nums: the average number of total
    Noting: img suffix must be .jpg
    """
    # enhance output path
    enhance_output_info_path = infopath
    if not box_offset:
        enhance_output_img_path = os.path.join(wdir, 'img')
        # imgpath = os.path.join(wdir, 'img')
    else:
        enhance_output_img_path = new_img_path
        # imgpath = new_img_path

    for enhance_cls in os.listdir(enhance_path):
        if os.path.isdir(os.path.join(enhance_path, enhance_cls)) and \
                len(os.listdir(os.path.join(enhance_path, enhance_cls))) != 0:

            total_times = int(
                avg_nums / len(os.listdir(os.path.join(enhance_path, enhance_cls))))   # 400 -> average number of cls
            if total_times >= 1:
                total_times = 1
            else:
                total_times = total_times
            # total_times = 10    # for one class
            print("Enhance class: {}, times: {}".format(enhance_cls, total_times))

            counter = 0
            for i in range(total_times):
                print("enhance {}: {}  times".format(enhance_cls, i))
                with tqdm(total=len(os.listdir(os.path.join(enhance_path, enhance_cls)))) as p_bar:
                    p_bar.set_description('enhance for: ' + enhance_cls)
                    for info_file in os.listdir(os.path.join(enhance_path, enhance_cls)):  # the least class
                        try:
                            name = info_file[:-4]
                            # print(name)
                            # Todo modify
                            if os.path.exists(os.path.join(enhance_output_img_path, name + '.jpg')):
                                image = cv2.imread(os.path.join(enhance_output_img_path, name + '.jpg'), -1)
                            else:
                                image = cv2.imread(os.path.join(enhance_output_img_path, name + '.png'), -1)
                            # print(image)
                            new_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                            # enhance_image = ImageEnhance.Brightness(new_image)
                            # bright = random.uniform(1.0, 1.0)   # setting range
                            # enhance_image = enhance_image.enhance(bright)

                            # enhance_image = ImageEnhance.Contrast(enhance_image)
                            # contrast = random.uniform(0.8, 1.2)     # setting range
                            # enhance_image = enhance_image.enhance(contrast)
                            # save image
                            # enhance_image.save(
                            #     os.path.join(enhance_output_img_path, 'enh_' + str(counter) + '_' + name + '.jpg'))
                            # # save info
                            # shutil.copy(os.path.join(wdir, 'enhance', enhance_cls, info_file),
                            #             os.path.join(enhance_output_info_path, 'enh_' + str(counter) + '_' + info_file))

                            enhance_contrast = contrastEnhancement(new_image)
                            enhance_contrast.save(os.path.join(enhance_output_img_path,
                                                  'enh_contrast_' + str(counter)+'_'+name + '.jpg'))
                            shutil.copy(os.path.join(enhance_path, enhance_cls, info_file),
                                        os.path.join(enhance_output_info_path, 'enh_contrast_'+str(counter)+'_'+info_file))

                            enhance_bright = brightnessEnhancement(new_image)
                            enhance_bright.save(os.path.join(enhance_output_img_path,
                                                  'enh_bright_' + str(counter)+'_'+name + '.jpg'))
                            shutil.copy(os.path.join(enhance_path, enhance_cls, info_file),
                                        os.path.join(enhance_output_info_path, 'enh_bright_'+str(counter)+'_'+info_file))

                            enhance_color = colorEnhancement(new_image)
                            enhance_color.save(os.path.join(enhance_output_img_path,
                                                  'enh_color_' + str(counter)+'_'+name + '.jpg'))
                            shutil.copy(os.path.join(enhance_path, enhance_cls, info_file),
                                        os.path.join(enhance_output_info_path, 'enh_color_'+str(counter)+'_'+info_file))

                            enhance_shape = sharp_enhance(new_image)
                            enhance_shape.save(os.path.join(enhance_output_img_path,
                                                  'enh_shape_' + str(counter)+'_'+name + '.jpg'))
                            shutil.copy(os.path.join(enhance_path, enhance_cls, info_file),
                                        os.path.join(enhance_output_info_path, 'enh_shape_'+str(counter)+'_'+info_file))

                            # enhance_noise = noise_enhance(new_image)
                            # enhance_noise.save(os.path.join(enhance_output_img_path,
                            #                       'enh_noise_' + str(counter)+'_'+name + '.jpg'))
                            # shutil.copy(os.path.join(enhance_path, enhance_cls, info_file),
                            #             os.path.join(enhance_output_info_path, 'enh_noise_'+str(counter)+'_'+info_file))
                        except Exception as e:
                            print(e)
                counter += 1


def generate_train_val_test_lst(wdir, train_ratio=0.8):
    """
    generate train,val,test text
    r = train_ratio
    train: val: test = r: (1-r)*r: (1-r)*(1-r)
    :param wdir:
    :param train_ratio: ratio of train set
    :return:
    """

    # infopath = os.path.join(wdir, 'info')   # img path
    # txtsavepath = os.path.join(wdir, 'ImageSets')
    # not_exists_path_make_dir(txtsavepath)


    train_percent = train_ratio   # train: r
    valtest_percent = 1 - train_ratio  # val + test: 1-r

    total_xml = os.listdir(infopath)
    num = len(total_xml)

    num_val_test = int(num * valtest_percent)  # val + test
    num_val = int(num_val_test * train_percent)      # val: (1-r)*r
    valtest = random.sample(range(num), num_val_test)  # num * num_val_test
    val = random.sample(valtest, num_val)  # num * num_val_test * r

    ftrain = open(os.path.join(lstpath, 'train.txt'), 'w')  # train
    # fvaltest = open(os.path.join(lstpath, 'valtest.txt'), 'w')  # val + test
    ftest = open(os.path.join(lstpath, 'test.txt'), 'w')    # test
    fval = open(os.path.join(lstpath, 'val.txt'), 'w')      # val

    print("make txt for train, val, test ...")
    for i in range(num):
        name = total_xml[i][:-4] + '\n'
        if i in valtest:
            # fvaltest.write(name)   # val + test
            if i in val:
                fval.write(name)   # val
            else:
                ftest.write(name)    # test
        else:
            ftrain.write(name)      # train

    # fvaltest.close()
    ftrain.close()
    fval.close()
    ftest.close()


def generate_dataset(wdir, gray_dataset=False, label_box=False):


    if label_box:
        imgpath = new_img_path

        img_train_path = os.path.join(dataset, 'images/train')
        label_train_path = os.path.join(dataset, 'labels/train')

        img_val_path = os.path.join(dataset, 'images/val')
        label_val_path = os.path.join(dataset, 'labels/val')

        img_test_path = os.path.join(dataset, 'images/test')
        label_test_path = os.path.join(dataset, 'labels/test')
    else:
        if gray_dataset:
            imgpath = os.path.join(wdir, 'gray')
        else:
            imgpath = os.path.join(wdir, 'img')

        # labelpath = os.path.join(wdir, 'info')
        # lstpath = os.path.join(wdir, 'ImageSets')

        # output path for train, val
        img_train_path = os.path.join(wdir, 'images/train')
        label_train_path = os.path.join(wdir, 'labels/train')

        img_val_path = os.path.join(wdir, 'images/val')
        label_val_path = os.path.join(wdir, 'labels/val')

        img_test_path = os.path.join(wdir, 'images/test')
        label_test_path = os.path.join(wdir, 'labels/test')

    # 按照划分好的训练文件的路径搜索目标，并将其复制到yolo格式下的新路径
    # train
    print("Making train data ...")
    copy_file(img_train_path, os.path.join(lstpath, 'train.txt'), imgpath)
    copy_file(label_train_path, os.path.join(lstpath, 'train.txt'), infopath)
    # val
    print("Making val data ...")
    copy_file(img_val_path, os.path.join(lstpath, 'val.txt'), imgpath)
    copy_file(label_val_path, os.path.join(lstpath, 'val.txt'), infopath)
    # testset
    print("Making test data ...")
    copy_file(img_test_path, os.path.join(lstpath, 'test.txt'), imgpath)
    copy_file(label_test_path, os.path.join(lstpath, 'test.txt'), infopath)


def copy_file(new_path, path_txt, search_path):
    # 参数1：存放新文件的位置  参数2：为上一步建立好的train,val训练数据的路径txt文件  参数3：为搜索的文件位置
    print(path_txt)
    not_exists_path_make_dir(new_path)
    with open(path_txt, 'r') as lines:
        filenames_to_copy = set(line.rstrip() for line in lines)
    for root, _, filenames in os.walk(search_path):
        for filename in filenames:
            if os.path.exists(os.path.join(new_path, filename)):
                continue
            else:
                if filename[:-4] + '.jpg' in filenames_to_copy or filename in filenames_to_copy or filename[:-4] in filenames_to_copy:
                    shutil.copy(os.path.join(root, filename), new_path)


def parse_args():
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--wdir', type=str, default="/cv/xyc/Datasets/antong/det/frontback", help="Dataset directory")
    parser.add_argument('--xml2info',     type=bool, default=False,
                        help="Read xml annotation and convert it to info")
    parser.add_argument('--all_visual', type=bool, default=False)
    parser.add_argument('--count_cls',    type=bool, default=False)
    parser.add_argument('--enhance',      type=bool, default=False)
    parser.add_argument('--gene_dataset', type=bool, default=True)
    parser.add_argument('--visualization',type=bool, default=False)
    parser.add_argument('--rgb2gray',     type=bool, default=False)
    parser.add_argument('--flag_gray',    type=bool, default=False, help="True: generate gray dataset; False: rgb dataset")
    parser.add_argument('--diff', type=bool, default=False, help="Compare the files in the two folders and move those two extra files")
    parser.add_argument('--data_mode',    type=int,  default=0, help="0 for general data, 1 for plate")
    parser.add_argument('--train_rate',    type=float,default=0.90)
    parser.add_argument('--test_number',  type=int,  default=20)
    parser.add_argument('--label_box', type=bool, default=True, help="label box transformation")
    parser.add_argument('--resize_size', type=int, nargs='+', default=[1280, 1920], help="resize size: width, height")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    opt = parse_args()
    wdir = opt.wdir
    sets = ['train', 'val', 'test']

    # Todo
    print(os.path.join(opt.wdir, ".."))
    # scp_command = 'scp -r ' + opt.wdir + ' 192.168.109.141:' + opt.wdir
    # print(scp_command)
    # os.system(scp_command)

    # 0. load labelmap
    jsonhander = JsonHandler(os.path.join(opt.wdir, "labelmap.json"))
    classes = jsonhander.load_json()['label']
    print(classes)
    print("nc:", len(classes))

    # Input
    imgpath = os.path.join(wdir, 'img')  # img path
    xmlpath = os.path.join(wdir, 'xml')  # xml path
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # output

    if opt.label_box:
        dataset = os.path.join(wdir, str(opt.resize_size[0]) + "x" + str(opt.resize_size[1]))
        new_img_path = os.path.join(dataset, "img")
        infopath = os.path.join(dataset, 'info')
        all_visual_path = os.path.join(dataset, 'all_visualization')  # 保存画框后图片的路径
        error_img_path = os.path.join(os.path.join(dataset, "error/img"))
        error_xml_path = os.path.join(os.path.join(dataset, "error/xml"))
        lstpath = os.path.join(dataset, 'ImageSets')
        enhance_path = os.path.join(dataset, "enhance")
        not_exists_path_make_dirs([new_img_path, infopath, all_visual_path, error_img_path, error_xml_path, lstpath])
    else:
        infopath = os.path.join(wdir, 'info')  # xml 2 info: [cls, x, y, w, h]
        not_exists_path_make_dir(infopath)
        all_visual_path = os.path.join(wdir, 'all_visualization')  # 保存画框后图片的路径
        error_img_path = os.path.join(os.path.join(wdir, "error/img"))
        error_xml_path = os.path.join(os.path.join(wdir, "error/xml"))
        enhance_path = os.path.join(wdir, "enhance")
        not_exists_path_make_dirs([error_img_path, error_xml_path])
    # 1. single img: xml, label, visual
    # read xml to info, write it to txt, and visual
    if opt.xml2info:
        xml2info(imgpath, xmlpath, mode_type=opt.data_mode, visual=opt.all_visual,
                 box_offset=opt.label_box, resize_size=opt.resize_size)  # mode_type: default=0, 1 for plate

    # 2. count class
    if opt.count_cls:   # count classes nums
        print("++++ count cls ++++")
        cls_nums = {k: 0 for k in classes}
        count_class_frequency(enhance_path, infopath, cls_nums, classes)
    # 3 enhance
    if opt.enhance and os.path.exists(enhance_path):
        count_the_number_of_directory = Count_the_number_of_directory(enhance_path)
        size, subdir_num, file_number, subdir_files_dict = count_the_number_of_directory.count()
        max_nums = max(subdir_files_dict.values())
        print("++++ enhance to max {} ++++".format(max_nums))
        enhance_img_xml(wdir, imgpath, 80, box_offset=opt.label_box)  # max_nums-1 don't for the cls of max_nums

    # 8 rgb2gray
    if opt.rgb2gray:
        rgb2gray_for_imgset_and_save(imgpath, os.path.join(wdir, "gray"))

    # 5 generate dataset
    if opt.gene_dataset:
        print("*--------- make dataset ----------*")
        generate_train_val_test_lst(wdir, opt.train_rate)
        generate_dataset(wdir, opt.flag_gray, opt.label_box)

    # 6 visulization
    if opt.visualization:
        train_val_test_visualization(wdir, colors)

    # 7 move these img don't labeled
    if opt.diff:
        print("filter xml ...")
        diff_img_lablefile_and_move_extra(infopath, xmlpath, os.path.join(wdir, "error/xml"), ".txt")    # 文件少的后缀
        print("filter img ...")
        diff_img_lablefile_and_move_extra(infopath, imgpath, os.path.join(wdir, "error/img"), ".txt")
    # TODO: add new dataset



