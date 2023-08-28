# encoding=utf-8
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

from convert import xyxy_convert_xywh, xywh_convert_xxyy

import sys

sys.path.append("../../../myscripts")
from my_io import JsonHandler, YamlHandler
from figure import draw_bar
from scripts_for_image import filter_different_suffix_with_the_same_name_of_2dirs, rgb2gray_for_imgset_and_save
from tools import not_exists_path_make_dir, not_exists_path_make_dirs
from file_operate import Count_the_number_of_directory, read_txt_and_return_list, save_file_according_suffixes
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

                        xmin, ymin, xmax, ymax = xywh_convert_xxyy((width, height),
                                                                   [float(x), float(y), float(w), float(h)])
                        # xmin = float(x) * float(width)
                        # ymin = float(y) * float(height)
                        # xmax = xmin + float(w) * float(width)
                        # ymax = ymin + float(h) * float(height)
                        # print(xmin, ymin, xmax, ymax)
                        labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                 colors[int(cls)], 1)
                        labelled = cv2.putText(labelled, classes[int(cls)], (int(xmin), int(ymin) - 2),
                                               cv2.FONT_HERSHEY_PLAIN, 2,
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
                            _new_jpg_file = os.path.join(wdir, "crop_str", classes[int(cls)],
                                                         file_name + str(i) + ".jpg")  # saved path
                            cv2.imwrite(_new_jpg_file, _cropImg)
                        except:
                            continue
                cv2.imwrite(os.path.join(retangele_img_path, file_name + '.jpg'), labelled)

                p_bar.update(1)


def train_val_test_visualization(wdir, colors):
    for set in sets:
        # Input
        imgs_path = os.path.join(wdir, 'images/%s' % (set))  # img path
        info_path = os.path.join(wdir, 'labels/%s' % (set))  # labels path

        # Output
        retangele_img_path = os.path.join(wdir, 'visualization/%s' % (set))  # 保存画框后图片的路径
        if not os.path.exists(retangele_img_path):
            os.makedirs(retangele_img_path)

        with tqdm(total=len(os.listdir(imgs_path))) as p_bar:
            p_bar.set_description('visual for train val test data')
            for file in os.listdir(imgs_path):
                name, suffix = os.path.splitext(file)
                # if len(file.split('.')) > 2:
                #     file_name = file[:-4]  # for file name have more than 1 dot
                # else:
                #     file_name = file.split('.')[0]
                #
                # if os.path.exists(os.path.join(imgs_path, file_name + '.png')):
                #     img_name = os.path.join(imgs_path, file_name + '.png')
                # elif os.path.exists(os.path.join(imgs_path, file_name + '.jpg')):
                #     img_name = os.path.join(imgs_path, file_name + '.jpg')
                # else:
                #     print(file_name)
                #     continue

                img = cv2.imread(os.path.join(imgs_path, file), -1)
                height, width = img.shape[0], img.shape[1]
                labelled = img

                with open(os.path.join(info_path, name + ".txt"), 'r') as label_info:
                    lines = label_info.readlines()
                    for i in range(len(lines)):
                        cls, x, y, w, h = lines[i].split(' ')

                        xmin, ymin, xmax, ymax = xywh_convert_xxyy((width, height),
                                                                   [float(x), float(y), float(w), float(h)])
                        # xmin = float(x) * float(width)
                        # ymin = float(y) * float(height)
                        # xmax = xmin + float(w) * float(width)
                        # ymax = ymin + float(h) * float(height)

                        labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                 colors[int(cls)], 2)
                        labelled = cv2.putText(labelled, classes[int(cls)], (int(xmin), int(ymin) - 2),
                                               cv2.FONT_HERSHEY_PLAIN, 2,
                                               colors[int(cls)], 2)
                cv2.imwrite(os.path.join(retangele_img_path, name + '.jpg'), labelled)

                p_bar.update(1)


def xml2info(imgpath, xmlpath, mode_type=0):
    """
    :param imgpath:
    :param xmlpath:
    :param mode_type: default=0, 1 for plate
    :return:
    """
    with tqdm(total=len(os.listdir(imgpath))) as p_bar:
        p_bar.set_description('xml2info')
        for file in os.listdir(imgpath):
            # file = "1a887921d97ed3260f03d6f7e968520.jpg"
            filename, suffix = os.path.splitext(file)
            if not os.path.exists(os.path.join(xmlpath, filename + ".xml")):
                continue
            check_orientation(file)
            continue
            parse_info = read_xml_to_lst(xmlpath, file, mode_type)

            if len(parse_info) == 0:
                print("The len(info)=0: ", file)
                shutil.move(os.path.join(imgpath, file), os.path.join(wdir, "error/img"))
                shutil.move(os.path.join(xmlpath, filename + ".xml"), os.path.join(wdir, "error/xml"))
                continue
            out_file = open(os.path.join(wdir, 'info', filename + '.txt'), 'w')
            for i in range(len(parse_info)):
                line = str(parse_info[i][0]) + " " \
                       + str(parse_info[i][1]) + " " \
                       + str(parse_info[i][2]) + " " \
                       + str(parse_info[i][3]) + " " \
                       + str(parse_info[i][4]) + "\n"
                out_file.write(line)
            out_file.close()

            p_bar.update(1)


def check_orientation(file):
    """
    判断多个矩形框的排列方向
    :param rects: 矩形框列表，每个元素格式为 ((x1,y1),(x2,y2))
    :return: 如果所有矩形框的横向长度之和大于等于纵向长度之和，则返回"横向排列"，否则返回"竖向排列"
    """
    image_id, suffix = os.path.splitext(file)
    in_file = open(os.path.join(xmlpath, '%s.xml' % (image_id)))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    bbox_xyxy = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls = cls.replace('/', '_')
        if cls not in classes:
            continue
        xmlbox = obj.find('bndbox')
        xmin, xmax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text)
        ymin, ymax = int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
        bbox_xyxy.append([xmin, ymin, xmax, ymax, cls])

    min_x = min(list(map(lambda x: x[0], bbox_xyxy)))
    min_y = min(list(map(lambda x: x[1], bbox_xyxy)))
    max_x = min(list(map(lambda x: x[2], bbox_xyxy)))
    max_y = min(list(map(lambda x: x[3], bbox_xyxy)))
    if (max_x - min_x) > 1.2 * (max_y - min_y) and (max_x - min_x) * (max_y - min_y) > 75000:
        # print("horizontal")
        shutil.move(os.path.join(xmlpath, image_id + ".xml"),
                    "/cv/all_training_data/control/tmp/horizontal/xml")
        shutil.move(os.path.join(imgpath, file),
                    "/cv/all_training_data/control/tmp/horizontal/img")
        shutil.move(os.path.join(wdir, "all_visualization", image_id + ".jpg"),
                    "/cv/all_training_data/control/tmp/horizontal/all_visualization")
    elif (max_y - min_y) > 1.2 * (max_x - min_x) and (max_x - min_x) * (max_y - min_y) > 75000:
        # print("vertical")
        shutil.move(os.path.join(xmlpath, image_id + ".xml"),
                    "/cv/all_training_data/control/tmp/vertical/xml")
        shutil.move(os.path.join(imgpath, file),
                    "/cv/all_training_data/control/tmp/vertical/img")
        shutil.move(os.path.join(wdir, "all_visualization", image_id + ".jpg"),
                    "/cv/all_training_data/control/tmp/vertical/all_visualization/")


def read_xml_to_lst(xmlpath, file, plate_mode_type):
    """
    read object info
    name -> cls
    bonx -> xmin, ymin, xmax, ymax
    plate_mode_type: default=0, 1 for plate
    return a lst [cls, xmin, ymin, xmax, ymax]
    """
    image_id, suffix = os.path.splitext(file)
    lst = []
    # fre_dict = {}
    in_file = open(os.path.join(xmlpath, '%s.xml' % (image_id)))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    if plate_mode_type == 0:  # for all data, except plate
        for obj in root.iter('object'):
            cls = obj.find('name').text
            cls = cls.replace('/', '_')

            if cls not in classes:
                continue
            xmlbox = obj.find('bndbox')
            xmin, xmax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text)
            ymin, ymax = int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
            try:
                newx, newy, neww, newh = xyxy_convert_xywh((w, h), [xmin, ymin, xmax, ymax])
                lst.append([classes.index(cls), newx, newy, neww, newh])
            except Exception as e:
                print("read_xml_to_lst:", image_id, e)

    else:  # for plate
        for obj in root.iter('object'):
            tmp_name = obj.find('name').text
            plate_type = tmp_name.split("/")[-1]
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
            else:
                # print("Error: ", image_id)
                continue

            xmlbox = obj.find('bndbox')
            xmin, xmax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text)
            ymin, ymax = int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
            try:
                newx, newy, neww, newh = xyxy_convert_xywh((w, h), [xmin, ymin, xmax, ymax])

                lst.append([classes.index(cls), newx, newy, neww, newh])
            except Exception as e:
                print("Error: ", e, image_id)
                continue

    # for obj in root.iter('object'):
    #     difficult = obj.find('difficult').text
    #     # TODO
    #     if plate_mode_type == 0:
    #         cls = obj.find('name').text
    #         cls = cls.replace('/', '_')
    #         if cls == "ao" or cls == "crack":
    #             break
    #         if cls not in classes:
    #             continue
    #     else:
    #         tmp_name = obj.find('name').text
    #         plate_type = tmp_name.split("/")[-1]
    #         if "/" in tmp_name:
    #
    #             # if plate_type == "WHITE":
    #             #     cls = "white_plate"
    #             # elif plate_type == "NE":
    #             #     cls = "black_plate"
    #             # elif plate_type == "NE":
    #             #     continue
    #             # else:
    #             #     cls = plate_type
    #             # if plate_type == "WHITE" or plate_type == "GREEN" or plate_type == "YELLOW":
    #             cls = "plate"
    #         else:
    #             print("Error: ", image_id)
    #             continue
    #
    #     xmlbox = obj.find('bndbox')
    #     xmin, xmax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text)
    #     ymin, ymax = int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
    #     try:
    #         newx, newy, neww, newh = xxyy_convert_xywh((w, h), [xmin, ymin, xmax, ymax])
    #
    #         lst.append([classes.index(cls), newx, newy, neww, newh])
    #     except:
    #         print("Error: ", image_id)
    #         continue

    return lst


def count_class_frequency(wdir, infopath, cls_nums, classes=None, flag_plot=True):
    for i in range(len(classes)):
        if not os.path.exists(os.path.join(wdir, 'enhance', classes[i])):
            os.makedirs(os.path.join(wdir, 'enhance', classes[i]))

    for file in os.listdir(infopath):
        f_info = open(os.path.join(infopath, file), 'r').readlines()
        try:
            for i in range(len(f_info)):
                cls = classes[int(f_info[i].split(" ")[0])]
                shutil.copy(os.path.join(infopath, file), os.path.join(wdir, 'enhance', cls))
                cls_nums.update([(cls, cls_nums[cls] + 1)])  # count subclass frequency
        except:
            print(file)

    print(cls_nums)
    class_frequency = open(os.path.join(wdir, 'enhance', 'class_frequency.txt'), 'w')
    class_frequency.write(json.dumps(cls_nums, indent=2))
    class_frequency.write('\n')
    class_frequency.close()

    if flag_plot:
        draw_bar(cls_nums, os.path.join(wdir, 'enhance'))


def enhance_img_xml(wdir, imgpath, avg_nums):
    """
    Aug: Brightness, contrast
    avg_nums: the average number of total
    Noting: img suffix must be .jpg
    """
    # enhance output path
    enhance_output_info_path = os.path.join(wdir, 'info')
    not_exists_path_make_dir(enhance_output_info_path)
    enhance_output_img_path = os.path.join(wdir, 'img')
    not_exists_path_make_dir(enhance_output_img_path)

    for enhance_cls in os.listdir(os.path.join(wdir, 'enhance')):
        # if enhance_cls.isdigit() or (enhance_cls >= 'A' and enhance_cls <= 'Z') \
        #         or (enhance_cls >= 'a' and enhance_cls <= 'z'):
        #     continue
        if os.path.isdir(os.path.join(wdir, 'enhance', enhance_cls)) and \
                len(os.listdir(os.path.join(wdir, 'enhance', enhance_cls))) != 0:
            total_times = int(avg_nums / len(os.listdir(os.path.join(wdir, 'enhance', enhance_cls)))) - 1  # 400 -> average number of cls
            if total_times >= 1:
                total_times = 1
            else:
                total_times = total_times
            # total_times = 10    # for one class
            print("Enhance class: {}, times: {}".format(enhance_cls, total_times))

            counter = 0
            for i in range(total_times):
                print("enhance {}: {}  times".format(enhance_cls, i))
                with tqdm(total=len(os.listdir(os.path.join(wdir, 'enhance', enhance_cls)))) as p_bar:
                    p_bar.set_description('enhance for: ' + enhance_cls)
                    for info_file in os.listdir(os.path.join(wdir, 'enhance', enhance_cls)):  # the least class
                        try:
                            name = info_file[:-4]
                            # print(name)
                            image = cv2.imread(os.path.join(imgpath, name + '.jpg'), -1)
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
                                                               'enh_contrast_' + str(counter) + '_' + name + '.jpg'))
                            shutil.copy(os.path.join(wdir, 'enhance', enhance_cls, info_file),
                                        os.path.join(enhance_output_info_path,
                                                     'enh_contrast_' + str(counter) + '_' + info_file))

                            enhance_bright = brightnessEnhancement(new_image)
                            enhance_bright.save(os.path.join(enhance_output_img_path,
                                                             'enh_bright_' + str(counter) + '_' + name + '.jpg'))
                            shutil.copy(os.path.join(wdir, 'enhance', enhance_cls, info_file),
                                        os.path.join(enhance_output_info_path,
                                                     'enh_bright_' + str(counter) + '_' + info_file))

                            enhance_color = colorEnhancement(new_image)
                            enhance_color.save(os.path.join(enhance_output_img_path,
                                                            'enh_color_' + str(counter) + '_' + name + '.jpg'))
                            shutil.copy(os.path.join(wdir, 'enhance', enhance_cls, info_file),
                                        os.path.join(enhance_output_info_path,
                                                     'enh_color_' + str(counter) + '_' + info_file))

                            enhance_shape = sharp_enhance(new_image)
                            enhance_shape.save(os.path.join(enhance_output_img_path,
                                                            'enh_shape_' + str(counter) + '_' + name + '.jpg'))
                            shutil.copy(os.path.join(wdir, 'enhance', enhance_cls, info_file),
                                        os.path.join(enhance_output_info_path,
                                                     'enh_shape_' + str(counter) + '_' + info_file))

                            # enhance_noise = noise_enhance(new_image)
                            # enhance_noise.save(os.path.join(enhance_output_img_path,
                            #                       'enh_noise_' + str(counter)+'_'+name + '.jpg'))
                            # shutil.copy(os.path.join(wdir, 'enhance', enhance_cls, info_file),
                            #             os.path.join(enhance_output_info_path, 'enh_noise_'+str(counter)+'_'+info_file))
                        except Exception as e:
                            print(e)
                counter += 1


def generate_train_val_test_lst(wdir, train_ratio=0.8):
    """
    Input: xml, img
    output: trai

    Generating txt for train, val, test set.
    Set = train + valtest(val + test)
        -train:    0.9
        -valtest: 0.1
            -- val:  0.1 * 0.1
            -- test: 0.1 * 0.9
    """

    infopath = os.path.join(wdir, 'info')  # img path
    txtsavepath = os.path.join(wdir, 'ImageSets')
    not_exists_path_make_dir(txtsavepath)

    train_percent = train_ratio  # train
    valtest_percent = 1 - train_ratio  # val + test

    total_xml = os.listdir(infopath)
    num = len(total_xml)

    num_val_test = int(num * valtest_percent)  # val + test
    tr = int(num_val_test * train_percent)  # val = num * 0.1 * 0.9
    valtest = random.sample(range(num), num_val_test)  # 从所有num中返回num_val_test个数量的项目
    val = random.sample(valtest, tr)  # val = num * 0.1 * 0.9

    ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')  # train
    fvaltest = open(os.path.join(txtsavepath, 'valtest.txt'), 'w')  # val + test
    ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
    fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')

    print("make txt for train, val, test ...")
    for i in range(num):
        # print(i)
        name = total_xml[i][:-4] + '\n'
        if i in valtest:
            fvaltest.write(name)  # val + test
            if i in val:
                fval.write(name)  # test
            else:
                ftest.write(name)  # val
        else:
            ftrain.write(name)  # train

    fvaltest.close()
    ftrain.close()
    fval.close()
    ftest.close()


def img_info_2_images_labels(wdir, gray_dataset=False):
    if gray_dataset:
        imgpath = os.path.join(wdir, 'gray')
    else:
        imgpath = os.path.join(wdir, 'img')
    labelpath = os.path.join(wdir, 'info')
    lstpath = os.path.join(wdir, 'ImageSets')

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
    copy_file(label_train_path, os.path.join(lstpath, 'train.txt'), labelpath)
    # val
    print("Making val data ...")
    copy_file(img_val_path, os.path.join(lstpath, 'val.txt'), imgpath)
    copy_file(label_val_path, os.path.join(lstpath, 'val.txt'), labelpath)
    # testset
    print("Making test data ...")
    copy_file(img_test_path, os.path.join(lstpath, 'test.txt'), imgpath)
    copy_file(label_test_path, os.path.join(lstpath, 'test.txt'), labelpath)


def copy_file(new_path, path_txt, search_path):
    # 参数1：存放新文件的位置  参数2：为上一步建立好的train,val训练数据的路径txt文件  参数3：为搜索的文件位置
    print(path_txt)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    with open(path_txt, 'r') as lines:
        filenames_to_copy = set(line.rstrip() for line in lines)
    for root, _, filenames in os.walk(search_path):
        for filename in filenames:
            if os.path.exists(os.path.join(new_path, filename)):
                continue
            else:
                if filename[:-4] + '.jpg' in filenames_to_copy or filename in filenames_to_copy or filename[
                                                                                                   :-4] in filenames_to_copy:
                    shutil.copy(os.path.join(root, filename), new_path)


def parse_args():
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--wdir', type=str, default="/cv/all_training_data/dangerous/clean/", help="Dataset directory")
    parser.add_argument('--xml2info', type=bool, default=False,
                        help="Read xml annotation and convert it to info")
    parser.add_argument('--count_cls', type=bool, default=False)
    parser.add_argument('--enhance', type=bool, default=False)
    parser.add_argument('--make_lst', type=bool, default=False)
    parser.add_argument('--gene_dataset', type=bool, default=False)
    parser.add_argument('--visualization', type=bool, default=False)
    parser.add_argument('--all_visual', type=bool, default=True)
    parser.add_argument('--rgb2gray', type=bool, default=False)
    parser.add_argument('--flag_gray', type=bool, default=False, help="True: generate gray dataset; False: rgb dataset")
    parser.add_argument('--diff', type=bool, default=False,
                        help="Compare the files in the two folders and move those two extra files")
    parser.add_argument('--data_mode', type=int, default=0, help="0 for general data, 1 for plate")
    parser.add_argument('--train_rate', type=float, default=0.90)
    parser.add_argument('--test_number', type=int, default=20)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    opt = parse_args()
    wdir = opt.wdir
    sets = ['train', 'val', 'test']

    # load labelmap
    jsonhander = JsonHandler(os.path.join(opt.wdir, "labelmap.json"))
    classes = jsonhander.load_json()['label']
    print(classes)
    print("nc:", len(classes))

    # Input
    imgpath = os.path.join(wdir, 'img')  # img path
    xmlpath = os.path.join(wdir, 'xml')  # xml path

    # output
    infopath = os.path.join(wdir, 'info')  # xml 2 info: [cls, x, y, w, h]
    not_exists_path_make_dir(infopath)

    # 1 xml 2 info, and write it to txt
    if opt.xml2info:
        print("++++ read *xml to info ++++")
        xml2info(imgpath, xmlpath, mode_type=opt.data_mode)  # mode_type: default=0, 1 for plate
    # tmp
    if opt.all_visual:
        print("++++ visualization for all img and xml ++++")
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
        labeled_visualization(wdir, colors)

    # 2 count class
    if opt.count_cls:  # count classes nums
        print("++++ count cls ++++")
        cls_nums = {k: 0 for k in classes}
        count_class_frequency(wdir, infopath, cls_nums, classes)

    # 3 enhance
    if opt.enhance and os.path.exists(os.path.join(wdir, 'enhance')):
        count_the_number_of_directory = Count_the_number_of_directory(os.path.join(wdir, 'enhance'))
        size, subdir_num, file_number, subdir_files_dict = count_the_number_of_directory.count()
        max_nums = max(subdir_files_dict.values())
        print("++++ enhance to max {} ++++".format(max_nums))
        enhance_img_xml(wdir, imgpath, 200)  # max_nums-1 don't for the cls of max_nums

    # 4 make txt
    if opt.make_lst:
        generate_train_val_test_lst(wdir, opt.train_rate)

    # 5 generate dataset
    if opt.gene_dataset:
        img_info_2_images_labels(wdir, opt.flag_gray)

    # 6 visulization
    if opt.visualization:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
        train_val_test_visualization(wdir, colors)

    # 7 move these img don't labeled
    if opt.diff:
        print("filter xml ...")
        diff_img_lablefile_and_move_extra(infopath, xmlpath, os.path.join(wdir, "error/xml"), ".txt")  # 文件少的后缀
        print("filter img ...")
        diff_img_lablefile_and_move_extra(infopath, imgpath, os.path.join(wdir, "error/img"), ".txt")
    # TODO: add new dataset

    # 8 rgb2gray
    if opt.rgb2gray:
        rgb2gray_for_imgset_and_save(imgpath, os.path.join(wdir, "gray"))
