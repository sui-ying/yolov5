#encoding=utf-8
import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import parse
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
from scripts_for_image import diff_img_lablefile_and_move_extra, rgb2gray_for_imgset_and_save
from tools import not_exists_path_make_dir, not_exists_path_make_dirs
from file_operate import Count_the_number_of_directory, split_img_labelfile_to_distpath, read_txt_and_return_list
from augment import contrastEnhancement, brightnessEnhancement, colorEnhancement, sharp_enhance, noise_enhance


class Process_label_and_generate_dataset():
    def __init__(self, kwargs):
        self.opt = kwargs

    # def _parse_args(self):
    #     parser = argparse.ArgumentParser(description='data preprocess')
    #     parser.add_argument('--wdir', type=str, default="/cv/xyc/Datasets/plate/taiguo/roi_det",
    #                         help="Dataset working directory")
    #     parser.add_argument('--split', type=bool, default=False, help="Files are saved in folders based on suffixes")
    #     parser.add_argument('--show_label_save', type=bool, default=False, help="show labels of the dataset and save")
    #     parser.add_argument('--xml2info', type=bool, default=False, help="Parse dataset annotation file")
    #     parser.add_argument('--all_visual', type=bool, default=True, help="visual for all dataset")
    #     parser.add_argument('--count_cls', type=bool, default=False, help="count the number of class")
    #     parser.add_argument('--enhance', type=bool, default=False, help="Enhance fewer data of class")
    #     parser.add_argument('--gene_dataset', type=bool, default=True, help="generate dataset")
    #     parser.add_argument('--visualization', type=bool, default=False, help="visualization for train,val,test")
    #     parser.add_argument('--rgb2gray', type=bool, default=False, help="rgb to gray")
    #     parser.add_argument('--flag_gray', type=bool, default=False,
    #                         help="True: generate gray dataset; False: rgb dataset")
    #     parser.add_argument('--diff', type=bool, default=False,
    #                         help="Compare the files in the two folders and move those two extra files")
    #     parser.add_argument('--data_mode', type=int, default=1, help="0 for general data, 1 for plate")
    #     parser.add_argument('--train_ratio', type=float, default=0.90, help="training set ratio")
    #     parser.add_argument('--test_number', type=int, default=20)
    #     parser.add_argument('--label_box', type=bool, default=False, help="label box transformation")
    #     parser.add_argument('--resize_size', type=int, nargs='+', default=[1280, 1920],
    #                         help="resize size: width, height")
    #     args = parser.parse_args()
    #     return args

    def _init_path(self):
        self.wdir = self.opt.wdir
        self.imgpath = os.path.join(self.wdir, 'img')  # img path
        self.xmlpath = os.path.join(self.wdir, 'xml')  # xml path
        self.sets = ['train', 'val', 'test']
        if self.opt.label_box:
            self.dataset = os.path.join(self.wdir, str(self.opt.resize_size[0]) + "x" + str(self.opt.resize_size[1]))
            self.new_img_path = os.path.join(self.dataset, "img")
            self.infopath = os.path.join(self.dataset, 'info')
            self.all_visual_path = os.path.join(self.dataset, 'all_visualization')  # 保存画框后图片的路径
            self.error_img_path = os.path.join(os.path.join(self.dataset, "error/img"))
            self.error_xml_path = os.path.join(os.path.join(self.dataset, "error/xml"))
            self.lstpath = os.path.join(self.dataset, 'ImageSets')
            self.enhance_path = os.path.join(self.dataset, "enhance")
            not_exists_path_make_dirs([self.new_img_path, self.infopath, self.all_visual_path, self.error_img_path,
                                       self.error_xml_path, self.lstpath, self.enhance_path])
        else:
            self.infopath = os.path.join(self.wdir, 'info')  # xml 2 info: [cls, x, y, w, h]
            self.all_visual_path = os.path.join(self.wdir, 'all_visualization')  # 保存画框后图片的路径
            self.error_img_path = os.path.join(os.path.join(self.wdir, "error/img"))
            self.error_xml_path = os.path.join(os.path.join(self.wdir, "error/xml"))
            self.lstpath = os.path.join(self.wdir, 'ImageSets')
            self.enhance_path = os.path.join(self.wdir, "enhance")
            not_exists_path_make_dirs([self.infopath, self.all_visual_path, self.error_img_path,
                                       self.error_xml_path, self.lstpath, self.enhance_path])

    def _load_labelmap(self):
        # load labelmap
        jsonhander = JsonHandler(os.path.join(self.opt.wdir, "labelmap.json"))
        classes = jsonhander.load_json()['label']
        print(classes)
        print("nc:", len(classes))
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
        self.classes = classes

    def _xml2info(self, visual=False, box_offset=False, resize_size=(1920, 1080), visual_cls=False):
        """
        :param mode_type: 0 for usual data, 1 for plate
        :param visual: visualization for image
        :param box_offset: flag for Calculate the bbox offset
        :param resize_size: (width, height) of resize size
        :return:
        """
        with tqdm(total=len(os.listdir(self.imgpath))) as p_bar:
            p_bar.set_description('xml2info')
            for file in os.listdir(self.imgpath):
                bbox_xywh = []
                img = cv2.imread(os.path.join(self.imgpath, file))
                height, width, channel = img.shape
                name, suffix = os.path.splitext(file)
                # 1. read xml to info
                if os.path.exists(os.path.join(self.xmlpath, name + ".xml")):
                    # lst_xyxy, parse_info = read_xml_to_lst(xmlpath, name, mode_type, box_offset=box_offset, resize_size=resize_size)
                    bbox_xyxy = self._read_xml_to_lst(name)
                else:
                    shutil.move(os.path.join(self.imgpath, file), self.error_img_path)
                    print(name + ".xml not exist")
                    p_bar.update(1)
                    continue
                if len(bbox_xyxy) == 0:
                    print("The len(info)=0: ", file)
                    shutil.move(os.path.join(self.imgpath, file), self.error_img_path)
                    shutil.move(os.path.join(self.xmlpath, name + ".xml"), self.error_xml_path)
                    p_bar.update(1)
                    continue

                # 2. img and box resize
                if box_offset:
                    # 对图像进行缩放并且进行长和宽的扭曲
                    img = cv2.resize(img, (resize_size[0], resize_size[1]), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(self.new_img_path, file), img)
                    # INTER_NEAREST - 最邻近插值
                    # INTER_LINEAR - 双线性插值，如果最后一个参数你不指定，默认使用这种方法
                    # INTER_CUBIC - 4x4像素邻域内的双立方插值
                    # INTER_LANCZOS4 - 8x8像素邻域内的Lanczos插值
                    bbox_xyxy = self._bbox_resize_nopad(width, height, bbox_xyxy, resize_size[0],
                                                        resize_size[1])  # np padding
                else:
                    w = width
                    h = height

                # 3. write label info to text
                # print(name, w, h)
                for i in range(len(bbox_xyxy)):
                    cls_index, xmin, ymin, xmax, ymax = bbox_xyxy[i]
                    newx, newy, neww, newh = xyxy_convert_xywh((w, h),
                                                               [float(xmin), float(ymin), float(xmax), float(ymax)])
                    bbox_xywh.append([cls_index, newx, newy, neww, newh])
                out_file = open(os.path.join(self.infopath, name + '.txt'), 'w')
                for i in range(len(bbox_xywh)):
                    line = str(bbox_xywh[i][0]) + " " \
                           + str(bbox_xywh[i][1]) + " " \
                           + str(bbox_xywh[i][2]) + " " \
                           + str(bbox_xywh[i][3]) + " " \
                           + str(bbox_xywh[i][4]) + "\n"
                    out_file.write(line)
                out_file.close()

                # 3. visual
                if visual and visual_cls:
                    visual_cls_path = os.path.join(os.path.join(self.wdir, "visualizaition_cls"))
                    for j in range(len(self.classes)):
                        not_exists_path_make_dir(os.path.join(visual_cls_path, self.classes[j]))

                    labelled = img
                    for i in range(len(bbox_xyxy)):
                        cls, xmin, ymin, xmax, ymax = bbox_xyxy[i]
                        labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                 self.colors[int(cls)], 2)
                        labelled = cv2.putText(labelled, self.classes[int(cls)], (int(xmin), int(ymin) - 2),
                                               cv2.FONT_HERSHEY_PLAIN,
                                               2, self.colors[int(cls)], 1)  # font scale, thickness
                    cv2.imwrite(os.path.join(self.all_visual_path, name + '.jpg'), labelled)
                    cv2.imwrite(os.path.join(visual_cls_path, self.classes[cls], name + '.jpg'), labelled)
                else:
                    labelled = img
                    for i in range(len(bbox_xyxy)):
                        cls, xmin, ymin, xmax, ymax = bbox_xyxy[i]
                        labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                 self.colors[int(cls)], 2)
                        labelled = cv2.putText(labelled, self.classes[int(cls)], (int(xmin), int(ymin) - 2),
                                               cv2.FONT_HERSHEY_PLAIN,
                                               2, self.colors[int(cls)], 1)  # font scale, thickness
                    cv2.imwrite(os.path.join(self.all_visual_path, name + '.jpg'), labelled)
                p_bar.update(1)

    def _read_xml_to_lst(self, image_id, plate_mode_type=0):
        """
        :param xmlpath: xml path
        :param image_id:
        :return: a 2d lst [[cls, xmin, ymin, xmax, ymax], [...]]
        """

        bbox_xyxy = []  # xyxy
        in_file = open(os.path.join(self.xmlpath, '%s.xml' % (image_id)))
        # in_file = open("/cv/xyc/dataset_yolov5/plate/dataset_5nc_model1st_20230306/error/xml/1647474178719_plate.xml")
        tree = ET.parse(in_file)
        root = tree.getroot()
        # size = root.find('size')
        # w = int(size.find('width').text)
        # h = int(size.find('height').text)

        for obj in root.iter('object'):
            cls = obj.find('name').text

            # all type
            if "/" in cls:  # for plate model1st
                cls = cls.split("/")[-1].split("_")[0].lower()
            if cls not in self.classes:
                continue

            # # taiguo model 1st
            # if len(cls) > 2:
            #     cls = "plate"
            # else:
            #     continue

            # print(cls)
            # if plate_mode_type == 0:  # for all data, except plate
            #     cls = obj.find('name').text
            #     cls = cls.replace('/', '_')
            #     if cls not in self.classes:
            #         continue
            #     # if cls == "ao":
            #     #     break
            #     # if len(cls) == 1:
            #     #     cls = cls.upper()
            #     # lorry
            #     # if cls == "tray-vertical" or cls == "tray":
            #     #     cls = "tray"
            #     # elif cls == "container-vertical" or cls == "container":
            #     #     cls = "container"
            #     # elif cls == "truckhead-down" or cls == "truckfront":
            #     #     cls = "truckfront"
            #     # elif cls == 'lorry-down' or cls == 'lorry-up' or cls == 'lorry':
            #     #     cls = "lorry"
            #     # elif cls == "truckhead-up" or cls == "truckback":
            #     #     cls = "truckback"
            #     # else:
            #     #     continue
            #     # print(cls)
            # else:  # for plate

            #     # 津ADA7550 / green_plate
            #     tmp_name = obj.find('name').text
            #     # for taiguo roi det
            #     # if len(tmp_name) > 2:
            #     #     # print(tmp_name)
            #     #     cls = "plate"
            #     # else:
            #     #     continue
            #     # for sanzahuo
            #     plate_type = tmp_name.split("/")[-1]
            #     # modify here
            #     if "/" in tmp_name:
            #         # sanzahuo
            #         if plate_type == "WHITE":
            #             cls = "white_plate"
            #         elif plate_type == "orange_plate":
            #             cls = "yellow_plate"
            #         elif plate_type == "NE":
            #             cls = "NE_plate"
            #         else:
            #             cls = plate_type
            #         # if plate_type == "WHITE" or plate_type == "GREEN" or plate_type == "YELLOW":
            #         # cls = "plate"

            xmlbox = obj.find('bndbox')
            xmin, xmax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text)
            ymin, ymax = int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
            bbox_xyxy.append([self.classes.index(cls), xmin, ymin, xmax, ymax])
        return bbox_xyxy

    def _bbox_resize_nopad(self, iw, ih, box, nw, nh):
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

    def _count_class_frequency(self, flag_plot=True, visual_cls=False):
        cls_nums = {k: 0 for k in self.classes}
        for i in range(len(self.classes)):
            if not os.path.exists(os.path.join(self.enhance_path, self.classes[i])):
                os.makedirs(os.path.join(self.enhance_path, self.classes[i]))

        for file in os.listdir(self.infopath):
            f_info = open(os.path.join(self.infopath, file), 'r').readlines()
            try:
                cls = self.classes[int(f_info[i].split(" ")[0])]
                shutil.copy(os.path.join(self.infopath, file), os.path.join(self.enhance_path, cls))
                cls_nums.update([(cls, cls_nums[cls] + 1)])  # count subclass frequency
            except Exception as e:
                print("_count_class_frequency:", e)

        print(cls_nums)
        class_frequency = open(os.path.join(self.enhance_path, 'class_frequency.txt'), 'w')
        class_frequency.write(json.dumps(cls_nums, indent=2))
        class_frequency.write('\n')
        class_frequency.close()

        if flag_plot:
            draw_bar(cls_nums, self.enhance_path)

    def _enhance_img_xml(self, avg_nums, box_offset=False):
        """
        Aug: Brightness, contrast
        avg_nums: the average number of total
        Noting: img suffix must be .jpg
        """
        # enhance output path
        enhance_output_info_path = self.infopath
        if not box_offset:
            enhance_output_img_path = os.path.join(self.wdir, 'img')
            # imgpath = os.path.join(wdir, 'img')
        else:
            enhance_output_img_path = self.new_img_path
            # imgpath = new_img_path

        for enhance_cls in os.listdir(self.enhance_path):
            if os.path.isdir(os.path.join(self.enhance_path, enhance_cls)) and \
                    len(os.listdir(os.path.join(self.enhance_path, enhance_cls))) != 0:

                total_times = int(
                    avg_nums / len(os.listdir(os.path.join(self.enhance_path, enhance_cls))))  # 400 -> average number of cls
                if total_times >= 1:
                    total_times = 1
                else:
                    total_times = total_times
                # total_times = 10    # for one class
                print("Enhance class: {}, times: {}".format(enhance_cls, total_times))

                counter = 0
                for i in range(total_times):
                    print("enhance {}: {}  times".format(enhance_cls, i))
                    with tqdm(total=len(os.listdir(os.path.join(self.enhance_path, enhance_cls)))) as p_bar:
                        p_bar.set_description('enhance for: ' + enhance_cls)
                        for info_file in os.listdir(os.path.join(self.enhance_path, enhance_cls)):  # the least class
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

                                enhance_contrast = contrastEnhancement(new_image)
                                enhance_contrast.save(os.path.join(enhance_output_img_path,
                                                                   'enh_contrast_' + str(
                                                                       counter) + '_' + name + '.jpg'))
                                shutil.copy(os.path.join(self.enhance_path, enhance_cls, info_file),
                                            os.path.join(enhance_output_info_path,
                                                         'enh_contrast_' + str(counter) + '_' + info_file))

                                enhance_bright = brightnessEnhancement(new_image)
                                enhance_bright.save(os.path.join(enhance_output_img_path,
                                                                 'enh_bright_' + str(counter) + '_' + name + '.jpg'))
                                shutil.copy(os.path.join(self.enhance_path, enhance_cls, info_file),
                                            os.path.join(enhance_output_info_path,
                                                         'enh_bright_' + str(counter) + '_' + info_file))

                                enhance_color = colorEnhancement(new_image)
                                enhance_color.save(os.path.join(enhance_output_img_path,
                                                                'enh_color_' + str(counter) + '_' + name + '.jpg'))
                                shutil.copy(os.path.join(self.enhance_path, enhance_cls, info_file),
                                            os.path.join(enhance_output_info_path,
                                                         'enh_color_' + str(counter) + '_' + info_file))

                                enhance_shape = sharp_enhance(new_image)
                                enhance_shape.save(os.path.join(enhance_output_img_path,
                                                                'enh_shape_' + str(counter) + '_' + name + '.jpg'))
                                shutil.copy(os.path.join(self.enhance_path, enhance_cls, info_file),
                                            os.path.join(enhance_output_info_path,
                                                         'enh_shape_' + str(counter) + '_' + info_file))

                                # enhance_noise = noise_enhance(new_image)
                                # enhance_noise.save(os.path.join(enhance_output_img_path,
                                #                       'enh_noise_' + str(counter)+'_'+name + '.jpg'))
                                # shutil.copy(os.path.join(self.enhance_path, enhance_cls, info_file),
                                #             os.path.join(enhance_output_info_path, 'enh_noise_'+str(counter)+'_'+info_file))
                            except Exception as e:
                                print(e)
                    counter += 1

    def _generate_train_val_test_lst(self):
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


        train_percent = self.opt.train_ratio   # train: r
        valtest_percent = 1 - self.opt.train_ratio  # val + test: 1-r

        total_xml = os.listdir(self.infopath)
        num = len(total_xml)

        num_val_test = int(num * valtest_percent)  # val + test
        num_val = int(num_val_test * train_percent)      # val: (1-r)*r
        valtest = random.sample(range(num), num_val_test)  # num * num_val_test
        val = random.sample(valtest, num_val)  # num * num_val_test * r

        ftrain = open(os.path.join(self.lstpath, 'train.txt'), 'w')  # train
        # fvaltest = open(os.path.join(lstpath, 'valtest.txt'), 'w')  # val + test
        ftest = open(os.path.join(self.lstpath, 'test.txt'), 'w')    # test
        fval = open(os.path.join(self.lstpath, 'val.txt'), 'w')      # val

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

    def _generate_dataset(self):

        if self.opt.label_box:
            imgpath = self.new_img_path
            img_train_path = os.path.join(self.dataset, 'images/train')
            label_train_path = os.path.join(self.dataset, 'labels/train')
            img_val_path = os.path.join(self.dataset, 'images/val')
            label_val_path = os.path.join(self.dataset, 'labels/val')
            img_test_path = os.path.join(self.dataset, 'images/test')
            label_test_path = os.path.join(self.dataset, 'labels/test')
        else:
            if self.opt.flag_gray:
                imgpath = os.path.join(self.wdir, 'gray')
            else:
                imgpath = os.path.join(self.wdir, 'img')

            # output path for train, val
            img_train_path = os.path.join(self.wdir, 'images/train')
            label_train_path = os.path.join(self.wdir, 'labels/train')
            img_val_path = os.path.join(self.wdir, 'images/val')
            label_val_path = os.path.join(self.wdir, 'labels/val')
            img_test_path = os.path.join(self.wdir, 'images/test')
            label_test_path = os.path.join(self.wdir, 'labels/test')

        # 按照划分好的训练文件的路径搜索目标，并将其复制到yolo格式下的新路径
        # train
        print("Making train data ...")
        self._copy_file(img_train_path, os.path.join(self.lstpath, 'train.txt'), imgpath)
        self._copy_file(label_train_path, os.path.join(self.lstpath, 'train.txt'), self.infopath)
        # val
        print("Making val data ...")
        self._copy_file(img_val_path, os.path.join(self.lstpath, 'val.txt'), imgpath)
        self._copy_file(label_val_path, os.path.join(self.lstpath, 'val.txt'), self.infopath)
        # testset
        print("Making test data ...")
        self._copy_file(img_test_path, os.path.join(self.lstpath, 'test.txt'), imgpath)
        self._copy_file(label_test_path, os.path.join(self.lstpath, 'test.txt'), self.infopath)

    def _copy_file(self, new_path, path_txt, search_path):
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
                    if filename[:-4] + '.jpg' in filenames_to_copy or filename in filenames_to_copy or filename[
                                                                                                       :-4] in filenames_to_copy:
                        shutil.copy(os.path.join(root, filename), new_path)

    def _train_val_test_visualization(self):
        if self.opt.label_box:
            datapath = self.dataset
        else:
            datapath = self.wdir

        for set in self.sets:
            # Input
            imgs_path = os.path.join(datapath, 'images/%s' % (set))  # img path
            info_path = os.path.join(datapath, 'labels/%s' % (set))  # labels path

            # Output
            retangele_img_path = os.path.join(datapath, 'visualization/%s' % (set))  # 保存画框后图片的路径
            not_exists_path_make_dir(retangele_img_path)

            with tqdm(total=len(os.listdir(imgs_path))) as p_bar:
                p_bar.set_description('visual for {} data'.format(set))
                for file in os.listdir(imgs_path):
                    name, suffix = os.path.splitext(file)
                    img = cv2.imread(os.path.join(imgs_path, file), -1)
                    height, width = img.shape[0], img.shape[1]
                    labelled = img

                    with open(os.path.join(info_path, name + ".txt"), 'r') as label_info:
                        lines = label_info.readlines()
                        for i in range(len(lines)):
                            cls, x, y, w, h = lines[i].split(' ')

                            xmin, ymin, xmax, ymax = xywh_convert_xxyy((width, height),
                                                                       [float(x), float(y), float(w), float(h)])
                            labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                                     self.colors[int(cls)], 2)
                            labelled = cv2.putText(labelled, self.classes[int(cls)], (int(xmin), int(ymin) - 2),
                                                   cv2.FONT_HERSHEY_PLAIN, 2, self.colors[int(cls)], 2)
                    cv2.imwrite(os.path.join(retangele_img_path, name + '.jpg'), labelled)
                    p_bar.update(1)

    def _show_label_and_count_class_frequence(self, mode=0):
        """
        :param mode: default 0; 1 for plate
        :return:
        """
        files = os.listdir(self.xmlpath)  # labeled file path
        classes = []
        fre_dict = {}

        # 遍历所有xml文件
        if mode == 1:  # for plate
            subclasses = []  # for a-z, 0-9
            sub_fre_dict = {}  # fre for subclassed
            with tqdm(total=len(files)) as p_bar:
                for xmlFile in files:
                    file_path = os.path.join(self.xmlpath, xmlFile)  # 第一个xml文件的路径
                    dom = parse(file_path)
                    root = dom.getroot()  # 获取根节点
                    for obj in root.iter('object'):  # 获取object节点中的name子节点
                        tmp_name = obj.find('name').text  # 找到object中 name 的具体名称

                        plate_type = tmp_name.split("/")[-1]
                        if "/" in tmp_name:
                            if plate_type not in classes:
                                classes.append(plate_type)
                                fre_dict[plate_type] = 1
                            else:
                                fre_dict.update([(plate_type, fre_dict[plate_type] + 1)])
                        else:
                            if plate_type not in subclasses:
                                subclasses.append(plate_type)
                                sub_fre_dict[plate_type] = 1
                            else:
                                sub_fre_dict.update([(plate_type, sub_fre_dict[plate_type] + 1)])
                    p_bar.update(1)

                print("class fre: ", fre_dict)
                print("subclass fre: ", sub_fre_dict)
                print("==== For second model: ")
                print("sorted subclasses: ", sorted(subclasses))
                print("len(subclasses): ", len(subclasses))

        else:
            with tqdm(total=len(files)) as p_bar:
                for xmlFile in files:
                    file_path = os.path.join(self.xmlpath, xmlFile)  # 第一个xml文件的路径
                    dom = parse(file_path)
                    root = dom.getroot()  # 获取根节点
                    for obj in root.iter('object'):  # 获取object节点中的name子节点
                        tmp_name = obj.find('name').text  # 找到object中 name 的具体名称
                        # tmp_name = tmp_name

                        # # for container
                        # if tmp_name == "zero" or tmp_name == "o":
                        #     tmp_name = "zero_O"
                        # elif tmp_name == "one" or tmp_name == "i":
                        #     tmp_name = "one_I"
                        # elif tmp_name == "two" or tmp_name == "z":
                        #     tmp_name = "two_Z"
                        # elif tmp_name == "five" or tmp_name == "s":
                        #     tmp_name = "five_s"
                        # elif tmp_name == "eight" or tmp_name == "b":
                        #     tmp_name = "eight_B"
                        # else:
                        #     tmp_name = tmp_name
                        # if tmp_name == "o":
                        #     print(xmlFile)
                        # else:
                        #     continue

                        if tmp_name not in classes:
                            classes.append(tmp_name)
                            fre_dict[tmp_name] = 1
                        else:
                            fre_dict.update([(tmp_name, fre_dict[tmp_name] + 1)])

                        # for container

                        # copy to distpath
                        # if tmp_name == "i":
                        #     try:
                        #         # print(xmlFile)
                        #         # shutil.copy(os.path.join(dist_xml_path, xmlFile), error_xml_path)
                        #         shutil.copy(os.path.join(dist_img_path, xmlFile[:-4] + ".jpg"), os.path.join(error_img_path, "i"))
                        #     except Exception as e:
                        #         print(e)
                        # if tmp_name == "z":
                        #     try:
                        #         # print(xmlFile)
                        #         # shutil.copy(os.path.join(dist_xml_path, xmlFile), error_xml_path)
                        #         shutil.copy(os.path.join(dist_img_path, xmlFile[:-4] + ".jpg"), os.path.join(error_img_path, "z"))
                        #     except Exception as e:
                        #         print(e)
                        # if tmp_name == "b":
                        #     try:
                        #         # print(xmlFile)
                        #         # shutil.copy(os.path.join(dist_xml_path, xmlFile), error_xml_path)
                        #         shutil.copy(os.path.join(dist_img_path, xmlFile[:-4] + ".jpg"), os.path.join(error_img_path, "b"))
                        #     except Exception as e:
                        #         print(e)
                        # if tmp_name == "s":
                        #     try:
                        #         # print(xmlFile)
                        #         # shutil.copy(os.path.join(dist_xml_path, xmlFile), error_xml_path)
                        #         shutil.copy(os.path.join(dist_img_path, xmlFile[:-4] + ".jpg"), os.path.join(error_img_path, "s"))
                        #     except Exception as e:
                        #         print(e)
                        # if tmp_name == "o":
                        #     try:
                        #         # print(xmlFile)
                        #         # shutil.copy(os.path.join(dist_xml_path, xmlFile), error_xml_path)
                        #         shutil.copy(os.path.join(dist_img_path, xmlFile[:-4] + ".jpg"), os.path.join(error_img_path, "o"))
                        #     except Exception as e:
                        #         print(e)
                    p_bar.update(1)

                _data = fre_dict.keys()
                classes.sort()

        print("sorted(classes): ", classes)
        print("len(classes): ", len(classes))
        print("Cls frequency: ", fre_dict)

        # write labelmap to json
        _dict = dict()
        _dict["label"] = classes
        jsonhander = JsonHandler(os.path.join(self.wdir, "labelmap.json"))
        jsonhander.save_json(_dict)

        with open(os.path.join(self.wdir, "origin_labelmap_fre.txt"), 'w') as fn:
            fn.write(json.dumps(classes) + "\n")
            fn.write(json.dumps(dict(sorted(fre_dict.items(), key=lambda x: x[0])), ensure_ascii=False))
        fn.close()

    def run(self):
        # split label and img
        self._init_path()
        # 1. split
        if self.opt.split:
            split_img_labelfile_to_distpath(self.wdir, self.wdir, ".xml")
        # 2. show label
        if self.opt.show_label_save:
            self._show_label_and_count_class_frequence(mode=self.opt.data_mode)  # 1 for plat
        if os.path.exists(os.path.join(self.wdir, "labelmap.json")):
            self._load_labelmap()
        # read xml to info, write it to txt, and visual
        if self.opt.xml2info:
            self._xml2info(visual=self.opt.all_visual, box_offset=self.opt.label_box,
                           resize_size=self.opt.resize_size, visual_cls=True)  # mode_type: default=0, 1 for plate

        # count class
        if self.opt.count_cls:  # count classes nums
            self._count_class_frequency()
        # enhance
        if self.opt.enhance and os.path.exists(self.enhance_path):
            count_the_number_of_directory = Count_the_number_of_directory(self.enhance_path)
            size, subdir_num, file_number, subdir_files_dict = count_the_number_of_directory.count()
            max_nums = max(subdir_files_dict.values())
            print("++++ enhance to max {} ++++".format(max_nums))
            self._enhance_img_xml(80, box_offset=self.opt.label_box)  # max_nums-1 don't for the cls of max_nums

        # rgb2gray
        # Todo, test
        if self.opt.rgb2gray:
            rgb2gray_for_imgset_and_save(self.imgpath, os.path.join(self.wdir, "gray"))

        # generate dataset
        if self.opt.gene_dataset:
            print("*--------- make dataset ----------*")
            self._generate_train_val_test_lst()
            self._generate_dataset()

        # visulization
        if self.opt.visualization:
            self._train_val_test_visualization()

        # move these img don't labeled
        # Todo: test
        if self.opt.diff:
            print("filter xml ...")
            diff_img_lablefile_and_move_extra(self.infopath, self.xmlpath, os.path.join(self.wdir, "error/xml"), ".txt")  # 文件少的后缀
            print("filter img ...")
            diff_img_lablefile_and_move_extra(self.infopath, self.imgpath, os.path.join(self.wdir, "error/img"), ".txt")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--wdir', type=str, default="/cv/xyc/Datasets/plate/taiguo/roi_det",
                        help="Dataset working directory")
    parser.add_argument('--split', type=bool, default=False, help="Files are saved in folders based on suffixes")
    parser.add_argument('--show_label_save', type=bool, default=False, help="show labels of the dataset and save")
    parser.add_argument('--xml2info', type=bool, default=False, help="Parse dataset annotation file")
    parser.add_argument('--all_visual', type=bool, default=True, help="visual for all dataset")
    parser.add_argument('--count_cls', type=bool, default=False, help="count the number of class")
    parser.add_argument('--enhance', type=bool, default=False, help="Enhance fewer data of class")
    parser.add_argument('--gene_dataset', type=bool, default=True, help="generate dataset")
    parser.add_argument('--visualization', type=bool, default=False, help="visualization for train,val,test")
    parser.add_argument('--rgb2gray', type=bool, default=False, help="rgb to gray")
    parser.add_argument('--flag_gray', type=bool, default=False,
                        help="True: generate gray dataset; False: rgb dataset")
    parser.add_argument('--diff', type=bool, default=False,
                        help="Compare the files in the two folders and move those two extra files")
    parser.add_argument('--data_mode', type=int, default=1, help="0 for general data, 1 for plate")
    parser.add_argument('--train_ratio', type=float, default=0.90, help="training set ratio")
    parser.add_argument('--test_number', type=int, default=20)
    parser.add_argument('--label_box', type=bool, default=False, help="label box transformation")
    parser.add_argument('--resize_size', type=int, nargs='+', default=[1280, 1920],
                        help="resize size: width, height")
    args = parser.parse_args()

    process_label_and_generate_dataset = Process_label_and_generate_dataset(args)
    process_label_and_generate_dataset.run()
