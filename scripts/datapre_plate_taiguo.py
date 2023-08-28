import xml.etree.ElementTree as ET
import pickle
import os
import shutil
import cv2
import random
import argparse

from tqdm import tqdm
import sys

from datapre import enhance_img_xml


# sys.path.append("../../../myscripts")
sys.path.append("/cv/gq/myscripts")
from my_io import JsonHandler
from figure import draw_bar
from tools import not_exists_path_make_dir, not_exists_path_make_dirs
from file_operate import Count_the_number_of_directory
from convert import xywh_convert_xxyy, xyxy_convert_xywh
from datapre import count_class_frequency

from scripts_for_image import rgb2gray_for_imgset_and_save
from file_operate import Count_the_number_of_directory
from augment import augment_det_dataset_img_info
from datapre import img_info_2_images_labels, generate_train_val_test_lst


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def delect_blank_and_rewrite(file):
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


def read_xml_to_lst(xmlpath, image_id):
    """
    read object info
    name -> cls
    bonx -> xmin, ymin, xmax, ymax
    return a lst [cls, xmin, ymin, xmax, ymax]
    """
    lst = []
    # dict = {}
    in_file =open(os.path.join(xmlpath, '%s.xml' % (image_id)))
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text   # 闽DG6236/yellow_plate

        if len(cls) > 2:  # 保留ROI
            cls = "plate/"
        # else:
        #     continue
        # else:
            # if "a" <= cls <= "z":
            #     cls = cls.upper()
                # cls = classes[origin_gt.index(cls)]  # for chinese labeled

        xmlbox = obj.find('bndbox')
        xmin, xmax = int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text)
        ymin, ymax = int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)
        lst.append([cls, xmin, ymin, xmax, ymax])  # for one plate
    return lst


def visualization_for_2nd_model_crop_img(wdir):

    imgs_path = os.path.join(wdir, "img")  # 图片路径
    labels_path = os.path.join(wdir, "info")  # info路径
    # Output
    visual_path = os.path.join(wdir, "visualization")
    visual_sf_path = os.path.join(wdir, "visualization_sf")  # for province
    not_exists_path_make_dirs([visual_path, visual_sf_path])

    tmp_label_error_path = os.path.join(wdir, "tmp_label_error")  #
    if not os.path.exists(tmp_label_error_path):
        os.makedirs(tmp_label_error_path)

    errorlst_2st = open(os.path.join(wdir, '2st_visual_error.txt'), 'w')

    for i in range(len(origin_gt)):
        not_exists_path_make_dir(os.path.join(visual_sf_path, origin_gt[i]))

    with tqdm(total=len(os.listdir(labels_path))) as p_bar:
        p_bar.set_description('visual for labeled data')
        for file in os.listdir(labels_path):
            try:
                file_name = file.split('.')[0]
                img = cv2.imread(os.path.join(imgs_path, file_name + '.jpg'), -1)
                height, width = img.shape[0], img.shape[1]
                labelled = img
                # with open(os.path.join(labels_path, file), 'r') as label_info:
                #     lines = label_info.readlines()
                # print(file_name)
                lines = delect_blank_and_rewrite(os.path.join(labels_path, file))
                _cls = ''
                sf = ''
                for i in range(len(lines)):
                    cls, x, y, w, h = lines[i].split(' ')
                    xmin, ymin, xmax, ymax = xywh_convert_xxyy((width, height), [x, y, w, h])
                    labelled = cv2.rectangle(labelled, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1)
                    # labelled = cv2.putText(labelled, cls, (int(xmin), int(ymax)), cv2.FONT_HERSHEY_PLAIN, 1,
                    #                                    (0, 255, 0), 1)
                    _cls += origin_gt[int(cls)]
                    if int(cls) > 33 and origin_gt[int(cls)] != "挂":   # 记录省份
                        sf = _cls
                cv2.imwrite(os.path.join(visual_sf_path, sf, file_name + "_" + _cls + '.jpg'), labelled)
                cv2.imwrite(os.path.join(visual_path, file_name + "_" + _cls + '.jpg'), labelled)
                p_bar.update(1)
            except:
                print("Error visualization: ", file)
                errorlst_2st.write(file + "\n")
                if os.path.exists(os.path.join(labels_path, file)) and \
                        not os.path.exists(os.path.join(tmp_label_error_path, file)):
                    shutil.move(os.path.join(labels_path, file), tmp_label_error_path)
                if os.path.exists(os.path.join(imgs_path, file_name + '.jpg')) and \
                    not os.path.exists(os.path.join(tmp_label_error_path, file_name + '.jpg')):
                    shutil.move(os.path.join(imgs_path, file_name + '.jpg'), tmp_label_error_path)


def parse_args():
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--datapre',      type=bool, default=False)
    parser.add_argument('--make_lst',     type=bool, default=True)
    parser.add_argument('--gene_dataset', type=bool, default=True)
    parser.add_argument('--all_visual',   type=bool, default=False)
    parser.add_argument('--rgb2gray',     type=bool, default=False)
    parser.add_argument('--count_cls',    type=bool, default=False)
    parser.add_argument('--enhance',      type=bool, default=False)
    parser.add_argument('--training_rate',type=float,default=0.9)
    parser.add_argument('--test_number',  type=int,  default=20)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    using for dataset_2nd: plate char recog
    """
    # print("lorry".upper())
    opt = parse_args()
    # img dataset
    wdir = "/cv/xyc/Datasets/plate/taiguo/roi_det"

    # plate dataset
    # crop_path = os.path.join(wdir, "../", "crop")
    crop_path = "/cv/xyc/Datasets/plate/taiguo/roi_recog"
    # crop_path = os.path.join(wdir, "crop")
    not_exists_path_make_dir(crop_path)

    remote_path = os.path.join(wdir, "..")
    print(remote_path)
    scp_command = 'scp -r ' + crop_path + ' 192.168.109.141:' + remote_path
    print(scp_command)
    os.system(scp_command)

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-',
               'p0', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']

    origin_gt = classes

    # classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    #            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    #            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    #            'U', 'V', 'W', 'X', 'Y', 'Z']

    _dict = {}
    _dict["label"] = classes
    jsonhander = JsonHandler(os.path.join(crop_path, "labelmap.json"))
    jsonhander.save_json(_dict)

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    imgpath = os.path.join(wdir, 'img')  # img path
    xmlpath = os.path.join(wdir, 'xml')  # xml path

    ## Output file for second model
    out_imgpath = os.path.join(crop_path, "img")  # save crop img
    out_txt_path = os.path.join(crop_path, "info")  # save label of crop img
    img_visualization = os.path.join(crop_path, "visualization")

    not_exists_path_make_dirs([out_imgpath, out_txt_path, img_visualization])

    if opt.datapre:
        # read xml, crop subimage, make labels
        for root, dirs, files in os.walk(xmlpath):
            with tqdm(total=len(files)) as p_bar:
                for file in files:
                    file_name = file.split('.')[0]
                    xml_file = os.path.join(xmlpath, file)  # xml file

                    if os.path.exists(os.path.join(imgpath, file_name + ".jpg")):
                        jpg_name = os.path.join(imgpath, file_name + ".jpg")
                    elif os.path.exists(os.path.join(imgpath, file_name + ".png")):
                        jpg_name = os.path.join(imgpath, file_name + ".png")
                    else:
                        continue
                    img = cv2.imread(jpg_name, -1)
                    height, width = img.shape[0], img.shape[1]   # HWC == BGR
                    labelled = img

                    # 1st step: read xml INFO
                    xml_info = read_xml_to_lst(xmlpath, file_name)
                    #####################################
                    # 按照单个车牌保存车牌以及其中的字符
                    plate_roi_lst = []
                    for i in range(len(xml_info)):
                        cls = xml_info[i][0]
                        if "/" in cls:
                            plate_roi_lst.append([xml_info[i]])

                    for j in range(len(xml_info)):
                        if "/" in xml_info[j][0]:
                            continue
                        else:
                            for jj in range(len(plate_roi_lst)):
                                x1, y1 = xml_info[j][1], xml_info[j][2]
                                # xmin < x1 < xmax and ymin < y1 < ymax
                                if x1 > plate_roi_lst[jj][0][1] and x1 < plate_roi_lst[jj][0][3] and \
                                   y1 > plate_roi_lst[jj][0][2] and y1 < plate_roi_lst[jj][0][4]:
                                    plate_roi_lst[jj].append(xml_info[j])
                    # plate_roi_lst = xml_info
                    # 2st crop subimg: plate
                    for ii in range(len(plate_roi_lst)):
                        single_plate = plate_roi_lst[ii]
                        for i in range(len(single_plate)):
                            cls, xmin, ymin, xmax, ymax = single_plate[i]
                            if "/" in cls:
                            # if cls == "LORRY":
                                cropImg = img[ymin: ymax, xmin: xmax]  # crop ymin: ymax, xmin: xmax
                                new_jpg_file = os.path.join(out_imgpath, file_name + "_" + str(ii) + ".jpg")  # saved path
                                cv2.imwrite(new_jpg_file, cropImg)

                    # 3rd save label to txt
                    for ii in range(len(plate_roi_lst)):
                        single_plate = plate_roi_lst[ii]
                        label_file = os.path.join(out_txt_path, file_name + "_" + str(ii) + '.txt')  # save label path
                        out_label_file = open(label_file, 'w')

                        for j in range(len(single_plate)):
                            if "/" in xml_info[j][0]:
                                _, subimg_xmin, subimg_ymin, subimg_xmax, subimg_ymax = single_plate[j]
                        # subimg_xmin, subimg_ymin, subimg_xmax, subimg_ymax = plate_roi_lst[ii][]
                        for i in range(len(single_plate)):
                            digit_cls, amin, bmin, amax, bmax = single_plate[i]
                            try:
                                if "/" not in digit_cls:
                                    x1, y1, w1, h1 = xyxy_convert_xywh((subimg_xmax - subimg_xmin, subimg_ymax - subimg_ymin),
                                                                       [amin - subimg_xmin, bmin - subimg_ymin, amax - subimg_xmin, bmax - subimg_ymin])
                                    out_label_file.write(
                                        str(origin_gt.index(digit_cls)) + " " + str(x1) + " " + str(y1) + " " + str(
                                            w1) + " " + str(h1) + "\n")
                                    # out_label_file.write(
                                    #     str(classes.index(digit_cls)) + " " + str(x1) + " " + str(y1) + " " + str(
                                    #         w1) + " " + str(h1) + "\n")
                            except Exception as e:
                                print(file_name, e)
                        out_label_file.close()

                    p_bar.update(1)

    # 4th visual
    if opt.all_visual:
        visualization_for_2nd_model_crop_img(crop_path)
    # print(num_danger)
    # # save bar figure
    # # draw_bar(num_danger, wdir)

    # 5 make txt
    if opt.make_lst:
        generate_train_val_test_lst(crop_path, opt.training_rate)

    # 6 generate dataset
    if opt.gene_dataset:
        img_info_2_images_labels(crop_path)

    # 7 rgb2gray
    if opt.rgb2gray:
        rgb2gray_for_imgset_and_save(os.path.join(crop_path, 'rgb'), out_imgpath)

    # 8 count class
    if opt.count_cls:   # count classes nums
        print("++++ count cls ++++")
        cls_nums = {k: 0 for k in classes}
        count_class_frequency(crop_path, out_txt_path, cls_nums, classes)

    # 8 enhance
    if opt.enhance and os.path.exists(os.path.join(crop_path, 'enhance')):
        count_the_number_of_directory = Count_the_number_of_directory(os.path.join(crop_path, 'enhance'))
        size, subdir_num, file_number, subdir_files_dict = count_the_number_of_directory.count()
        max_nums = max(subdir_files_dict.values())
        print("++++ enhance to max {} ++++".format(max_nums))
        enhance_img_xml(crop_path, out_imgpath, 200)  # max_nums-1 don't for the cls of max_nums