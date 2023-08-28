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

sys.path.append("../../../myscripts")
from my_io import JsonHandler, YamlHandler
from figure import draw_bar
from scripts_for_image import filter_different_suffix_with_the_same_name_of_2dirs, rgb2gray_for_imgset_and_save
from tools import not_exists_path_make_dir, not_exists_path_make_dirs
from file_operate import move_dir_to_another_dir, copy_dir_to_another_dir
from file_operate import Count_the_number_of_directory, read_txt_and_return_list, save_file_according_suffixes
from augment import contrastEnhancement, brightnessEnhancement, colorEnhancement, sharp_enhance, noise_enhance


from process_and_gene_dataset import Process_label_and_generate_dataset


class Process_and_expand_dataset(Process_label_and_generate_dataset):
    """
    using for expand dataset
    """
    def __init__(self, args):
        super().__init__(args)
        self.opt = args

    # 方法重写
    def _init_path(self):
        self.wdir = self.opt.wdir
        self.imgpath = os.path.join(self.wdir, 'img')  # img path
        self.xmlpath = os.path.join(self.wdir, 'xml')  # xml path
        not_exists_path_make_dirs([self.imgpath, self.xmlpath])
        self.sets = ['train', 'val', 'test']

        self.infopath = os.path.join(self.wdir, 'info')  # xml 2 info: [cls, x, y, w, h]
        self.all_visual_path = os.path.join(self.wdir, 'all_visualization')  # 保存画框后图片的路径

        self.error_img_path = os.path.join(self.wdir, "error/img")
        self.error_xml_path = os.path.join(self.wdir, "error/xml")
        self.enhance_path = os.path.join(self.wdir, "enhance")
        # target
        self.target = self.opt.target_path
        self.target_imgpath = os.path.join(self.target, 'img')  # img path
        self.target_xmlpath = os.path.join(self.target, 'xml')  # xml path
        self.target_infopath = os.path.join(self.target, 'info')  # xml 2 info: [cls, x, y, w, h]
        self.target_all_visual_path = os.path.join(self.target, 'all_visualization')  # 保存画框后图片的路径
        self.lstpath = os.path.join(self.target, 'ImageSets')

    def _copy_source_to_target(self):

        # img
        copy_dir_to_another_dir(self.imgpath, self.target_imgpath)

        # xml
        copy_dir_to_another_dir(self.xmlpath, self.target_xmlpath)

        # info
        copy_dir_to_another_dir(self.infopath, self.target_infopath)

        # all_visual
        copy_dir_to_another_dir(self.all_visual_path, self.target_all_visual_path)

    def _move_source_to_target(self):

        # img
        move_dir_to_another_dir(self.imgpath, self.target_imgpath)

        # xml
        move_dir_to_another_dir(self.xmlpath, self.target_xmlpath)

        # info
        move_dir_to_another_dir(self.infopath, self.target_infopath)

        # all_visual
        move_dir_to_another_dir(self.all_visual_path, self.target_all_visual_path)

    def _generate_train_val_test_lst(self):
        """
        generate train,val,test text
        r = train_ratio
        train: val: test = r: (1-r)*r: (1-r)*(1-r)
        :param wdir:
        :param train_ratio: ratio of train set
        :return:
        """

        train_percent = self.opt.train_ratio   # train: r
        valtest_percent = 1 - self.opt.train_ratio  # val + test: 1-r

        total_xml = os.listdir(self.target_infopath)
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

        # output path for train, val
        img_train_path = os.path.join(self.target, 'images/train')
        label_train_path = os.path.join(self.target, 'labels/train')
        img_val_path = os.path.join(self.target, 'images/val')
        label_val_path = os.path.join(self.target, 'labels/val')
        img_test_path = os.path.join(self.target, 'images/test')
        label_test_path = os.path.join(self.target, 'labels/test')

        # 按照划分好的训练文件的路径搜索目标，并将其复制到yolo格式下的新路径
        # train
        print("Making train data ...")
        self._copy_file(img_train_path, os.path.join(self.lstpath, 'train.txt'), self.target_imgpath)
        self._copy_file(label_train_path, os.path.join(self.lstpath, 'train.txt'), self.target_infopath)
        # val
        print("Making val data ...")
        self._copy_file(img_val_path, os.path.join(self.lstpath, 'val.txt'), self.target_imgpath)
        self._copy_file(label_val_path, os.path.join(self.lstpath, 'val.txt'), self.target_infopath)
        # testset
        print("Making test data ...")
        self._copy_file(img_test_path, os.path.join(self.lstpath, 'test.txt'), self.target_imgpath)
        self._copy_file(label_test_path, os.path.join(self.lstpath, 'test.txt'), self.target_infopath)

    def _load_labelmap(self):
        # load labelmap
        jsonhander = JsonHandler(os.path.join(self.target, "labelmap.json"))
        classes = jsonhander.load_json()['label']
        print(classes)
        print("nc:", len(classes))
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
        self.classes = classes

    def run(self):
        # source(wdir) -> target

        # 0. init
        self._init_path()

        # 1. split
        if len(os.listdir(self.imgpath)) == 0 and len(os.listdir(self.xmlpath)) == 0:
            save_file_according_suffixes(self.wdir)
        not_exists_path_make_dirs([self.infopath, self.all_visual_path, self.error_img_path,
                                   self.error_xml_path, self.lstpath, self.enhance_path])

        # 1. load labelmap
        if os.path.exists(os.path.join(self.target, "labelmap.json")):
            self._show_label_and_count_class_frequence(mode=self.opt.data_mode)  # 1 for plate
            self._load_labelmap()
        else:
            self._show_label_and_count_class_frequence(mode=self.opt.data_mode)  # 1 for plate
            self._load_labelmap()

        # 2. read xml to info, write it to txt, and visual
        self._xml2info(visual=self.opt.all_visual, box_offset=self.opt.label_box,
                       resize_size=self.opt.resize_size, visual_cls=False)  # mode_type: default=0, 1 for plate

        # 3. move source(img, xml, info, all_visual) to target_path(wdir)
        if self.opt.merge:
            self._move_source_to_target()

        # 4. generate datasets
        if self.opt.gene_dataset:
            # rm -rf target images, labels
            if os.path.exists(os.path.join(self.target, "images")):
                shutil.rmtree(os.path.join(self.target, "images"))
            if os.path.exists(os.path.join(self.target, "labels")):
                shutil.rmtree(os.path.join(self.target, "labels"))

            print("*--------- make dataset ----------*")
            self._generate_train_val_test_lst()
            self._generate_dataset()

        # 5 rm empty self.wdir
        shutil.rmtree(self.wdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--wdir', type=str, default="/cv/xyc/Datasets/taiguo_yancan/labeled/0428/检测/Side2_subimg",
                        help="Source dataset directory")
    parser.add_argument('--target_path', type=str, default="/cv/all_training_data/yancan/xinhaida/det/side",
                        help="target dataset directory")
    parser.add_argument('--xml2info', type=bool, default=True, help="Parse dataset annotation file")
    parser.add_argument('--all_visual', type=bool, default=True, help="visual for all dataset")
    parser.add_argument('--count_cls', type=bool, default=False, help="count the number of class")
    parser.add_argument('--enhance', type=bool, default=False, help="Enhance fewer data of class")
    parser.add_argument('--gene_dataset', type=bool, default=True, help="generate dataset")
    parser.add_argument('--merge', type=bool, default=True, help="expand target_path")
    parser.add_argument('--visualization', type=bool, default=False, help="visualization for train,val,test")
    parser.add_argument('--rgb2gray', type=bool, default=False, help="rgb to gray")
    parser.add_argument('--flag_gray', type=bool, default=False,
                        help="True: generate gray dataset; False: rgb dataset")
    parser.add_argument('--diff', type=bool, default=False,
                        help="Compare the files in the two folders and move those two extra files")
    parser.add_argument('--data_mode', type=int, default=0, help="0 for general data, 1 for plate")
    parser.add_argument('--train_ratio', type=float, default=0.90, help="training set ratio")
    parser.add_argument('--test_number', type=int, default=20)
    parser.add_argument('--label_box', type=bool, default=False, help="label box transformation")
    parser.add_argument('--resize_size', type=int, nargs='+', default=[1280, 1920],
                        help="resize size: width, height")
    args = parser.parse_args()

    process_and_expand_dataset = Process_and_expand_dataset(args)
    process_and_expand_dataset.run()