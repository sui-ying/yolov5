import json
import os
import os.path
import shutil
from xml.etree.ElementTree import parse, Element
from tqdm import tqdm
import sys
import argparse

sys.path.append("../../../myscripts")

from my_io import JsonHandler, YamlHandler
from file_operate import split_img_labelfile_to_distpath, read_txt_and_return_list
from tools import not_exists_path_make_dir, not_exists_path_make_dirs


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


def show_label_and_count_class_frequence(source_path, mode=0):
    """
    :param source_path: show xml label, and count class frequence
    :param mode: default 0; 1 for plate
    :return:
    """

    dist_img_path = os.path.join(source_path, "img")
    dist_xml_path = os.path.join(source_path, "xml")
    not_exists_path_make_dir(dist_img_path)
    not_exists_path_make_dir(dist_img_path)

    error_xml_path = os.path.join(source_path, "error", "xml")
    error_img_path = os.path.join(source_path, "error", "img")
    not_exists_path_make_dir(error_xml_path)
    not_exists_path_make_dir(error_img_path)

    files = os.listdir(dist_xml_path)  # labeled file path
    cnt = 0
    classes = []
    fre_dict = {}

    # 遍历所有xml文件
    if mode == 1:  # for plate
        subclasses = []  # for a-z, 0-9
        sub_fre_dict = {}  # fre for subclassed
        with tqdm(total=len(files)) as p_bar:
            for xmlFile in files:
                file_path = os.path.join(dist_xml_path, xmlFile)  # 第一个xml文件的路径
                dom = parse(file_path)
                root = dom.getroot()  # 获取根节点
                for obj in root.iter('object'):  # 获取object节点中的name子节点
                    tmp_name = obj.find('name').text  # 找到object中 name 的具体名称\

                    plate_type = tmp_name.split("/")[-1]
                    # if plate_type == "green_plate":
                    # 	shutil.move(os.path.join(dist_xml_path, xmlFile)),
                    # 	shutil.move(dist_img_path, )
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

            # print(classes)
            # print(sorted(classes))
            # print(len(classes))
            print("class fre: ", fre_dict)
            print("subclass fre: ", sub_fre_dict)

            print("==== For second model: ")
            print("sorted subclasses: ", sorted(subclasses))
            print("len(subclasses): ", len(subclasses))

    else:  # for all model
        # not_exists_path_make_dirs([os.path.join(error_img_path, "i"), os.path.join(error_img_path, "z"),
        #                            os.path.join(error_img_path, "b"), os.path.join(error_img_path, "s"),
        #                            os.path.join(error_img_path, "o")])
        with tqdm(total=len(files)) as p_bar:
            for xmlFile in files:
                file_path = os.path.join(dist_xml_path, xmlFile)  # 第一个xml文件的路径
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
    # print("classes: ", classes)

    print("sorted(classes): ", classes)
    print("len(classes): ", len(classes))
    print("Cls frequency: ", fre_dict)


    _dict = {}
    _dict["label"] = classes
    jsonhander = JsonHandler(os.path.join(source_path, "labelmap.json"))
    jsonhander.save_json(_dict)

    with open(os.path.join(source_path, "origin_labelmap_fre.txt"), 'w') as fn:
        fn.write(json.dumps(classes) + "\n")
        fn.write(json.dumps(dict(sorted(fre_dict.items(), key=lambda x: x[0])), ensure_ascii=False))
    fn.close()
    # write json
    # _class = {"label": {}}
    # with open(os.path.join(source_path, "label_map.json"), 'w') as f:
    #     json.dump(classes, f, indent=2)
    # f.close()


def check_det_error_label_and_save(wdir):
    """
    img, xml, info --> error/xml, img, info
    :param wdir:
    :return:
    """

    # wdir = "/cv/xyc/Datasets/xinhaida_v1_2/xinhaidav2_side_tiexiu_liewen_20221202"
    img_path = os.path.join(wdir, "img")
    xml_path = os.path.join(wdir, "xml")
    label_path = os.path.join(wdir, "info")

    error_label_dir = os.path.join(wdir, "error")
    error_img_path = os.path.join(error_label_dir, "img")
    error_json_path = os.path.join(error_label_dir, "xml")
    error_label_path = os.path.join(error_label_dir, "info")

    not_exists_path_make_dirs([error_img_path, error_json_path, error_label_path])

    error_txt = os.path.join(wdir, "error_labeled.txt")
    error_lst = read_txt_and_return_list(error_txt)
    for i in range(len(error_lst)):
        file = error_lst[i]
        filename, _ = os.path.splitext(file)
        # print(file)

        # img
        img_suffix = [".png", ".jpg"]
        for j in range(len(img_suffix)):
            if os.path.exists(os.path.join(img_path, filename + img_suffix[j])):
                imgname = filename + img_suffix[j]
                # print(imgname)
                if not os.path.exists(os.path.join(error_img_path, imgname)):
                    shutil.move(os.path.join(img_path, imgname), error_img_path)
                break

        # json
        if os.path.exists(os.path.join(xml_path, filename+".xml")) and not os.path.exists(os.path.join(error_json_path, filename+".xml")):
            shutil.move(os.path.join(xml_path, filename+".xml"), error_json_path)

        # label
        if os.path.exists(os.path.join(label_path, filename+".txt")) and not os.path.exists(os.path.join(error_label_path, filename+".txt")):
            shutil.move(os.path.join(label_path, filename+".txt"), error_label_path)


def parse_args():
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--wdir', type=str, default="/cv/xyc/Datasets/antong/det/container", help="Dataset directory")
    parser.add_argument('--split', type=bool, default=True, help="Read xml annotation and convert it to info")
    parser.add_argument('--show_label_save', type=bool, default=True)
    parser.add_argument('--data_mode', type=int, default=0, help="0 for general data, 1 for plate")
    parser.add_argument('--test_rate', type=float, default=0.1)
    parser.add_argument('--test_number', type=int,  default=20)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    opt = parse_args()
    source_path = opt.wdir

    ori_data_path = source_path
    dist_path = source_path
    # dist_path = "/cv/xyc/Datasets/xinhaida_v1_2/det/top"

    if opt.split:
        # 1. split
        split_img_labelfile_to_distpath(ori_data_path, dist_path, ".xml")

    if opt.show_label_save:
        # 2. show label
        show_label_and_count_class_frequence(dist_path, mode=opt.data_mode)  # 1 for plate

    # check_det_error_label_and_save(source_path)