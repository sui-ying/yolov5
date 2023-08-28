import xml.etree.ElementTree as ET
import pickle
import os
import shutil
import cv2
import random
import argparse
import numpy as np
import json
from PIL import Image
from PIL import ImageEnhance

from tqdm import tqdm
import sys
sys.path.append("../../myscripts")
from my_io import JsonHandler
from figure import draw_bar
from tools import not_exists_path_make_dir, not_exists_path_make_dirs
from scripts_for_image import filter_different_suffix_with_the_same_name_of_2dirs
from convert import xywh_convert_xxyy, xyxy_convert_xywh, xxyy2xywh

from figure import draw_bar
from file_operate import Count_the_number_of_directory, save_file_according_suffixes
from augment import contrastEnhancement, brightnessEnhancement, colorEnhancement, sharp_enhance, noise_enhance
from process_and_gene_dataset import Process_label_and_generate_dataset


class Generate_ROI_Recog_Dataset(Process_label_and_generate_dataset):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # for project_type
        if self.args.project_type == "Thailand":
            self.origin_gt = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-',
                              'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']
            self.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-',
                              'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']

        elif self.args.project_type == "PSA":
            self.origin_gt = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                              "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                              "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                              "U", "V", "W", "X", "Y", "Z"]
            self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                              "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                              "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                              "U", "V", "W", "X", "Y", "Z"]

        elif self.args.project_type == "type":
            self.origin_gt = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                              "a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n",
                              "o", "p", "q", "r", "s", "t", "u", "w", "x", "y", "z"]
            self.classes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                              "a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n",
                              "o", "p", "q", "r", "s", "t", "u", "w", "x", "y", "z"]
        else:
            print("**** Warning project_type is None !")
            self.origin_gt = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                              "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                              "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                              "W", "X", "Y", "Z", "沪", "苏", "豫", "皖", "浙", "京",
                              "津", "渝", "冀", "滇", "辽", "黑", "湘", "鲁", "川", "赣",
                              "鄂", "甘", "晋", "陕", "吉", "闽", "黔", "粤", "桂", "青",
                              "琼", "宁", "藏", "蒙", "新", "学", "挂", "警", "港", "内",
                              "-"]

            self.classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                       'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k',
                       'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v',
                       'w', 'x', 'y', 'z', 'sh', 'js', 'ha', 'ah', 'zj', 'bj',
                       'tj', 'cq', 'he', 'yn', 'ln', 'hl', 'hn', 'sd', 'sc', 'jx',
                       'hb', 'gs', 'sx', 'sn', 'jl', 'fj', 'gz', 'gd', 'gx', 'qh',
                       'hi', 'nx', 'xz', 'nm', 'xj', 'xue', 'gua', 'jiing', 'gang', 'inside',
                       '-']

        self.wdir = self.args.wdir
        self.imgpath = os.path.join(self.wdir, 'img')  # img path
        self.xmlpath = os.path.join(self.wdir, 'xml')  # xml path
        self.crop_path = self.args.crop_path
        not_exists_path_make_dir(self.crop_path)

        _dict = {}
        _dict["label"] = self.classes
        jsonhander = JsonHandler(os.path.join(self.crop_path, "labelmap.json"))
        jsonhander.save_json(_dict)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.classes))]

        # Output file for second model
        self.out_imgpath = os.path.join(self.crop_path, "img")  # save crop img
        self.infopath = os.path.join(self.crop_path, "info")  # save label of crop img
        self.img_visualization = os.path.join(self.crop_path, "visualization")
        self.error_path = os.path.join(self.crop_path, "error")
        self.error_img = os.path.join(self.error_path, "img")
        self.error_xml = os.path.join(self.error_path, "xml")
        self.enhance_path = os.path.join(self.crop_path, "enhance")
        self.lstpath = os.path.join(self.crop_path, 'ImageSets')
        self.sets = ['train', 'val', 'test']
        not_exists_path_make_dirs([self.out_imgpath, self.infopath, self.img_visualization,
                                   self.enhance_path, self.lstpath, self.error_img, self.error_xml])
        if self.args.flag_gray:
            self.gray_img = os.path.join(self.crop_path, "gray_img")
            not_exists_path_make_dir(self.gray_img)

    def _delect_blank_and_rewrite(self, file):
        with open(file, 'r') as f1:
            lines = f1.read().splitlines()
            while '' in lines:
                lines.remove('')
        f1.close()

        with open(file, 'w') as f2:
            if len(lines) < 2:
                f2.write(lines)
            else:
                for i in range(len(lines) - 1):
                    f2.write(lines[i] + "\n")
                f2.write(lines[-1])
        f2.close()
        return lines

    def _read_xml_to_lst(self, image_id):
        """
        read object info
        name -> cls
        bonx -> xmin, ymin, xmax, ymax
        return a lst [cls, xmin, ymin, xmax, ymax]
        """
        lst = []
        in_file = open(os.path.join(self.xmlpath, image_id + ".xml"))
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text

            if self.args.project_type == "dongguan":  # for cn
                # dongguan
                if cls not in self.args.origin_gt and '/' not in cls:
                    continue
                if '/' in cls:  # 保留ROI
                    cls = cls
                else:
                    if "a" <= cls <= "z":
                        cls = cls.upper()
            elif self.args.project_type == "Thailand":  # Thailand
                if len(cls) > 2:  # 保留ROI
                    cls = "ROI"
            elif self.args.project_type == "qqctu":

                if len(cls) > 2 and "/" in cls:
                    cls = "ROI"
                elif cls == "-" or cls not in self.args.origin_gt:
                    continue
            elif self.args.project_type == "type" or self.args.project_type == "letter" or self.args.project_type == "digit":
                # labelmap
                if cls in ["wordh", "numhl", "numhs", "wordv", "numvl", "numvs", "numhsq"]:
                    cls = "ROI"
            else:
                if len(cls) > 2:
                    cls = "ROI"

            xmlbox = obj.find('bndbox')
            xmin, xmax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text)
            ymin, ymax = int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
            lst.append([cls, xmin, ymin, xmax, ymax])  # for one plate
        return lst

    def _count_class_frequency(self, flag_plot=True, visual_cls=False):
        cls_nums = {k: 0 for k in self.classes}
        for i in range(len(self.classes)):
            if not os.path.exists(os.path.join(self.enhance_path, self.classes[i])):
                os.makedirs(os.path.join(self.enhance_path, self.classes[i]))

        for file in os.listdir(self.infopath):
            f_info = open(os.path.join(self.infopath, file), 'r').readlines()
            try:
                for j in range(len(f_info)):
                    cls = self.classes[int(f_info[j].split(" ")[0])]
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
            enhance_output_img_path = self.out_imgpath
            # imgpath = os.path.join(wdir, 'img')
        else:
            enhance_output_img_path = self.new_img_path
            # imgpath = new_img_path

        for enhance_cls in os.listdir(self.enhance_path):
            if enhance_cls.isdigit():
                continue
            if os.path.isdir(os.path.join(self.enhance_path, enhance_cls)) and \
                    len(os.listdir(os.path.join(self.enhance_path, enhance_cls))) != 0:

                # 400 -> average number of cls
                total_times = int(avg_nums / len(os.listdir(os.path.join(self.enhance_path, enhance_cls))))
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

                                enhance_noise = noise_enhance(new_image)
                                enhance_noise.save(os.path.join(enhance_output_img_path,
                                                                'enh_noise_' + str(counter) + '_' + name + '.jpg'))
                                shutil.copy(os.path.join(self.enhance_path, enhance_cls, info_file),
                                            os.path.join(enhance_output_info_path,'enh_noise_' + str(counter) + '_' + info_file))
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
        train_percent = self.args.train_ratio   # train: r
        valtest_percent = 1 - self.args.train_ratio  # val + test: 1-r

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

        if self.args.label_box:
            imgpath = self.new_img_path
            img_train_path = os.path.join(self.dataset, 'images/train')
            label_train_path = os.path.join(self.dataset, 'labels/train')
            img_val_path = os.path.join(self.dataset, 'images/val')
            label_val_path = os.path.join(self.dataset, 'labels/val')
            img_test_path = os.path.join(self.dataset, 'images/test')
            label_test_path = os.path.join(self.dataset, 'labels/test')
        else:
            if self.args.flag_gray:
                imgpath = os.path.join(self.crop_path, 'gray_img')
            else:
                imgpath = os.path.join(self.crop_path, 'img')

            # output path for train, val
            img_train_path = os.path.join(self.crop_path, 'images/train')
            label_train_path = os.path.join(self.crop_path, 'labels/train')
            img_val_path = os.path.join(self.crop_path, 'images/val')
            label_val_path = os.path.join(self.crop_path, 'labels/val')
            img_test_path = os.path.join(self.crop_path, 'images/test')
            label_test_path = os.path.join(self.crop_path, 'labels/test')

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
        """
        :param new_path: 存放新文件的位置
        :param path_txt: 为上一步建立好的train,val训练数据的路径txt文件
        :param search_path: 为搜索的文件位置
        :return:
        """
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
            datapath = self.crop_path

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
                                                     (0, 0, 255), 2)
                            labelled = cv2.putText(labelled, self.classes[int(cls)], (int(xmin), int(ymin) - 2),
                                                   cv2.FONT_HERSHEY_PLAIN, 2, self.colors[int(cls)], 2)
                    cv2.imwrite(os.path.join(retangele_img_path, name + '.jpg'), labelled, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                    p_bar.update(1)

    def run(self):

        # 1. split
        if not os.path.exists(self.imgpath) or not os.path.exists(self.xmlpath):
            not_exists_path_make_dirs([self.imgpath, self.xmlpath])
            if len(os.listdir(self.imgpath)) == 0 and len(os.listdir(self.xmlpath)) == 0:
                save_file_according_suffixes(self.wdir)

        # read xml, crop subimage, make labels
        if self.args.datapre:
            for root, dirs, files in os.walk(self.imgpath):
                with tqdm(total=len(files)) as p_bar:
                    for file in files:
                        try:
                            file_name, suffix = os.path.splitext(file)
                            xml_file = os.path.join(self.xmlpath, file_name + ".xml")  # xml file
                            if not os.path.exists(xml_file):
                                continue
                            img = cv2.imread(os.path.join(self.imgpath, file), -1)
                            height, width = img.shape[0], img.shape[1]  # HWC == BGR

                            # 1st step: read xml INFO
                            xml_info = self._read_xml_to_lst(file_name)
                            plate_roi_lst = [[x] for x in xml_info if x[0] == "ROI"]

                            # Match the ROI and the characters in it
                            for j in range(len(xml_info)):
                                if xml_info[j][0] == "ROI":
                                    continue
                                for jj in range(len(plate_roi_lst)):
                                    x1, y1 = xml_info[j][1], xml_info[j][2]
                                    # xmin < x1 < xmax and ymin < y1 < ymax
                                    if x1 + 2 > plate_roi_lst[jj][0][1] and x1 + 2 < plate_roi_lst[jj][0][3] and \
                                       y1 + 2 > plate_roi_lst[jj][0][2] and y1 + 2 < plate_roi_lst[jj][0][4]:
                                        plate_roi_lst[jj].append(xml_info[j])
                            # Horizontal sort
                            plate_roi_lst = [sorted(d, key=lambda x: x[1]) for d in plate_roi_lst]

                            # 2st crop subimg: plate
                            for ii in range(len(plate_roi_lst)):
                                # filter: only label plate roi
                                if len(plate_roi_lst[ii]) == 1:
                                    continue
                                single_plate = plate_roi_lst[ii][0]
                                cls, xmin, ymin, xmax, ymax = single_plate

                                if cls != "ROI":
                                    print(file_name, "label data error")
                                    # label false data
                                    for j in range(len(plate_roi_lst)):
                                        for i in range(len(plate_roi_lst[j])):
                                            if plate_roi_lst[j][i][0] == 'ROI':
                                                plate_roi_lst[j][0], plate_roi_lst[j][i] = plate_roi_lst[j][i], plate_roi_lst[j][0]
                                                break
                                    single_plate = plate_roi_lst[ii][0]
                                    cls, xmin, ymin, xmax, ymax = single_plate

                                cropImg = img[ymin: ymax, xmin: xmax]  # crop ymin: ymax, xmin: xmax
                                new_jpg_file = os.path.join(self.out_imgpath, file_name + "_" + str(ii) + ".jpg")  # saved path
                                cv2.imwrite(new_jpg_file, cropImg)

                                # save gray roi img
                                if self.args.flag_gray:
                                    gray_image = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
                                    new_jpg_file = os.path.join(self.gray_img,
                                                                file_name + "_" + str(ii) + ".jpg")  # saved path
                                    cv2.imwrite(new_jpg_file, gray_image)

                            # 3rd save label to txt
                            for ii in range(len(plate_roi_lst)):
                                # crop subimg: plate
                                cls, subimg_xmin, subimg_ymin, subimg_xmax, subimg_ymax = plate_roi_lst[ii][0]
                                cropImg = img[subimg_ymin: subimg_ymax, subimg_xmin: subimg_xmax]  # crop ymin: ymax, xmin: xmax
                                cropImg = np.pad(cropImg, ((0, 20), (0, 0), (0, 0)), mode='constant',
                                                 constant_values=(255, 255))

                                single_plate = plate_roi_lst[ii][1:]
                                label_file = os.path.join(self.infopath, file_name + "_" + str(ii) + '.txt')
                                out_label_file = open(label_file, 'w')

                                plate_char = ""
                                for i in range(len(single_plate)):

                                    digit_cls, amin, bmin, amax, bmax = single_plate[i]
                                    x1, y1, w1, h1 = xyxy_convert_xywh((subimg_xmax - subimg_xmin, subimg_ymax - subimg_ymin),
                                                    [amin - subimg_xmin, bmin - subimg_ymin, amax - subimg_xmin,
                                                     bmax - subimg_ymin])
                                    out_label_file.write(str(self.classes.index(digit_cls)) + " " + str(x1) + " "
                                                          + str(y1) + " " + str(w1) + " " + str(h1) + "\n")

                                    # xmin, ymin, xmax, ymax = xywh_convert_xxyy((subimg_xmax - subimg_xmin, subimg_ymax - subimg_ymin),
                                    #                                            [float(x1), float(y1), float(w1), float(h1)])

                                    if self.args.all_visual:
                                        cv2.rectangle(cropImg, (int(amin - subimg_xmin), int(bmin - subimg_ymin)),
                                                                 (int(amax - subimg_xmin), int(bmax - subimg_ymin)),
                                                                 (0, 0, 255), 1)
                                        plate_char += digit_cls
                                if self.args.all_visual:
                                    cv2.putText(cropImg, plate_char, (10, int(subimg_ymax - subimg_ymin) + 20),
                                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)  # font scale, thickness
                                    cv2.imwrite(os.path.join(self.img_visualization, file_name + "_" + str(ii) + '.jpg'), cropImg,
                                                            [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                                out_label_file.close()
                            p_bar.update(1)

                        except Exception as E:
                            print(file_name, E)

        # count cls
        if self.args.count_cls:  # count classes nums
            self._count_class_frequency()
        # enhance
        if self.args.enhance and os.path.exists(self.enhance_path):
            count_the_number_of_directory = Count_the_number_of_directory(self.enhance_path)
            size, subdir_num, file_number, subdir_files_dict = count_the_number_of_directory.count()
            max_nums = max(subdir_files_dict.values())
            print("++++ enhance to max {} ++++".format(max_nums))
            self._enhance_img_xml(max_nums)  # max_nums-1 don't for the cls of max_nums

        if self.args.gene_dataset:
            print("*--------- make dataset ----------*")
            self._generate_train_val_test_lst()
            self._generate_dataset()

        # visulization
        if self.args.visualization:
            self._train_val_test_visualization()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--project_type', type=str, default="type", help='detection type: Thailand, PSA, plate')
    parser.add_argument('--wdir', type=str,
                        default="/cv/all_training_data/container/roi/dataset1",
                        help="source img path, for roi detect(model1st) dataset")
    parser.add_argument('--crop_path', type=str,
                        default="/cv/all_training_data/container/type/dataset1",
                        help="sub img path, for roi recog(model2nd) dataset")
    parser.add_argument('--datapre', type=bool, default=True)
    parser.add_argument('--all_visual', type=bool, default=True)
    parser.add_argument('--visualization', type=bool, default=False)
    parser.add_argument('--count_cls', type=bool, default=False, help="count the number of class")
    parser.add_argument('--enhance', type=bool, default=False, help="Enhance fewer data of class")
    parser.add_argument('--gene_dataset', type=bool, default=True)
    parser.add_argument('--flag_gray', type=bool, default=False,
                        help="True: generate gray dataset; False: rgb dataset")
    parser.add_argument('--label_box', type=bool, default=False, help="label box transformation")
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_number', type=int, default=20)
    args = parser.parse_args()

    generate_ROI_Recog_Dataset = Generate_ROI_Recog_Dataset(args)
    generate_ROI_Recog_Dataset.run()

    """
    used for generating ROI_recog dataset, such as plate, lorry recog
    Get to start:
        modify:
            wdir,
            crop_path, 
            project_type
    """