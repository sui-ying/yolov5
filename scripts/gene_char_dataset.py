import numpy as np
import os
import argparse
import sys
import cv2
import random
from tqdm import tqdm

sys.path.append("../../myscripts")
from tools import not_exists_path_make_dir, not_exists_path_make_dirs
from convert import xywh_convert_xxyy
from my_io import JsonHandler, Txthandler


class Generate_Char_Dataset():

    def __init__(self, args):
        self.args = args
        self.wdir = self.args.wdir
        self.img_path = os.path.join(self.wdir, "img")
        self.lbl_path = os.path.join(self.wdir, "info")
        self.labelmap_path = os.path.join(self.wdir, "labelmap.json")
        if os.path.exists(self.labelmap_path):
            jsonhander = JsonHandler(self.labelmap_path)
            self.labelmap = jsonhander.load_json()['label']
        else:
            return

        self.labelmap_gt = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B",
                        "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P",
                        "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "沪", "苏",
                        "豫", "皖", "浙", "京", "津", "渝", "冀", "滇", "辽", "黑", "湘", "鲁",
                        "川", "赣", "鄂", "甘", "晋", "陕", "吉", "闽", "黔", "粤", "桂", "青",
                        "琼", "宁", "藏", "蒙", "新", "学", "挂", "警", "港", "内", '-']
        self.char_path = os.path.join(self.wdir, "char.txt")

    ######################3

    def _get_ordered_result(self, digit_location, recog_tag):
        """
        get all prediction and order them in corresponding order
        :param digit_location: a list of two dimension array, the length of list is
                               the number of classes. each array in this list
                               represents the region in each class. e.g. list[0] is
                               background regions and list[1] is first class regions.
                               list[1] is a two-dimension array,which each row is a
                               region.each region has five elements: x1,y1,x2,y2,score
        :param recog_tag: a int, label of roi
        :return:
            ordered_result: a two dimension numpy array, size is x*6. x is the number
                            of detected region, each ros is: label,x1,y1,x2,y2,score
        """
        ordered_result = np.zeros((0, 6))
        sub_result = np.zeros((1, 6))

        for i in range(1, len(digit_location)):
            location = digit_location[i]
            if location.shape[0] == 0:
                continue
            for j in range(location.shape[0]):
                sub_result[0, 0] = i
                sub_result[0, 1:6] = location[j, 0:5]
                ordered_result = np.concatenate((ordered_result, sub_result), 0)

        if self.detect_type == 'char' or self.detect_type == 'truck_num':
            sortindex = ordered_result[:, 3].argsort()
            ordered_result = ordered_result[sortindex, :]
            if self.detect_type == 'char':
                ordered_result, is_two_row = self._check_two_row_plate(ordered_result)
        else:
            if recog_tag == 6 or recog_tag == 5 or recog_tag == 4:
                if self._image_transfer == 'fpga' or self.use_fpga:
                    sortindex = ordered_result[:, 3].argsort()
                else:
                    sortindex = ordered_result[:, 2].argsort()
                ordered_result = ordered_result[sortindex, :]
            if recog_tag == 3 or recog_tag == 2 or recog_tag == 1:
                sortindex = ordered_result[:, 3].argsort()
                ordered_result = ordered_result[sortindex, :]
            if recog_tag == 7 and ordered_result.size > 0:
                ordered_result = self._sort_muilti_raw_result(ordered_result)
        return ordered_result

    def _sort_muilti_raw_result(self, raw_result):
        """
        sort multi-row digit result
        :param raw_result:
        :return:
        """
        sorted_index = raw_result[:, 2].argsort()
        raw_result = raw_result[sorted_index]
        final_result = np.zeros((0, raw_result.shape[1]))
        index = 1
        base_y = raw_result[0, 2]
        letter_height = np.mean(raw_result[:, 4] - raw_result[:, 2])
        each_row_first_id = [0]
        while index < raw_result.shape[0]:
            if raw_result[index, 2] - base_y > 0.7 * letter_height:
                each_row_first_id.append(index)
                base_y = raw_result[index, 2]
            index += 1

        for i in range(len(each_row_first_id)):
            if i < len(each_row_first_id) - 1:
                tmp_result = raw_result[each_row_first_id[i]:each_row_first_id[i + 1]]
                sorted_index = tmp_result[:, 3].argsort()
                tmp_result = tmp_result[sorted_index]
                final_result = np.concatenate((final_result, tmp_result), 0)
            else:
                tmp_result = raw_result[each_row_first_id[i]:]
                sorted_index = tmp_result[:, 3].argsort()
                tmp_result = tmp_result[sorted_index]
                final_result = np.concatenate((final_result, tmp_result), 0)
        return final_result

    #######################


    def _check_two_row_plate(self, label):
        """
        check whether the plate has two rows.
        if it has two rows, return ordered result
        :param label: a x*6 two-dimension numpy array, x is the number of detected
                      region,each row is: label,x1,y1,x2,y2,score
        :return:
        """
        # sortindex = label[:, 3].argsort()
        # label = label[sortindex, :]
        if label.shape[0]==0:
            return label,False
        final_label= np.zeros((0,6))
        firstrow_index=[]
        secondrow_index=[]
        row_index=0
        while True:
            next_row=row_index+1
            length_current=label[row_index,4]-label[row_index,2]
            while True:
                if next_row==label.shape[0]:
                    secondrow_index.append(row_index)
                    break
                length_next=label[next_row,4]-label[next_row,2]
                if (label[row_index,2]-label[next_row,2])>0.9*length_next:
                    firstrow_index.append(next_row)
                    next_row=next_row+1
                elif (label[next_row,2]-label[row_index,2])>0.9*length_current:
                    firstrow_index.append(row_index)
                    row_index = next_row
                    break
                else:
                    secondrow_index.append(row_index)
                    row_index=next_row
                    break
            if row_index==label.shape[0]-1:
                secondrow_index.append(row_index)
                break
            if next_row==label.shape[0]:
                break

        if len(secondrow_index)==label.shape[0]:
            return label, False

        for i in firstrow_index:
            final_label=np.concatenate((final_label,label[i:i+1,:]),0)
        for i in secondrow_index:
            final_label = np.concatenate((final_label, label[i:i + 1, :]), 0)

        return final_label, True

    def _sort_muilti_bbox(self, bbox):
        """
        sort multi-row digit result
        :param bbox: nparray, label,x1,y1,x2,y2,score
        :return:
        """
        sorted_index = bbox[:, 2].argsort()
        bbox = bbox[sorted_index]
        final_result = np.zeros((0, bbox.shape[1]))
        index = 1
        base_y = bbox[0, 2]
        letter_height = np.mean(bbox[:, 4] - bbox[:, 2])
        each_row_first_id = [0]
        while index < bbox.shape[0]:
            if bbox[index, 2] - base_y > 0.6 * letter_height:  #
                each_row_first_id.append(index)
                base_y = bbox[index, 2]
            index += 1

        for i in range(len(each_row_first_id)):
            if i < len(each_row_first_id) - 1:
                tmp_result = bbox[each_row_first_id[i]:each_row_first_id[i + 1]]
                sorted_index = tmp_result[:, 3].argsort()
                tmp_result = tmp_result[sorted_index]
                final_result = np.concatenate((final_result, tmp_result), 0)
            else:
                tmp_result = bbox[each_row_first_id[i]:]
                sorted_index = tmp_result[:, 3].argsort()
                tmp_result = tmp_result[sorted_index]
                final_result = np.concatenate((final_result, tmp_result), 0)
        return final_result

    def run(self):

        char_file = open(self.char_path, 'w', encoding='utf-8')
        with tqdm(total=len(os.listdir(self.img_path))) as p_bar:
            p_bar.set_description("parser voc label to crnn data")
            files = os.listdir(self.img_path)
            random.shuffle(files)
            for file in files:
                filename, suffix = os.path.splitext(file)
                # filename = "enh_shape_0_CK9出闸挂车牌相机_ck9_20230507140129_20230507140135_142767548_0_077_150_0"
                img_file = os.path.join(self.img_path, file)
                lbl_fiel = os.path.join(self.lbl_path, filename + ".txt")

                img = cv2.imread(img_file)
                H, W, C = img.shape

                txthandler = Txthandler(lbl_fiel)
                infos = txthandler.read_txt()

                # parser label to bbox
                bbox = []
                for i in range(len(infos)):
                    _bbox = infos[i].split(" ")
                    x1, y1, x2, y2 = xywh_convert_xxyy((W, H), _bbox[1:])
                    bbox.append([_bbox[0], str(x1), str(y1), str(x2), str(y2)])

                bbox_array = np.array(bbox).astype(np.float32)
                sorted_bbox = self._sort_muilti_bbox(bbox_array)    # for double plate

                labels = [self.labelmap_gt[int(bbox[0])] for bbox in sorted_bbox]
                char = "".join("{}".format(label) for label in labels)

                if int(sorted_bbox[0][0]) < 33:
                    print(char, filename)
                    cv2.imshow("demo", img)
                    cv2.waitKey()

                char_file.write(file + " " + char + "\n")
                p_bar.update(1)

        char_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preprocess')
    parser.add_argument('--wdir', type=str, default="/cv/all_training_data/plate/haikou_plate_20230531/subimg/",
                        help="Dataset working directory")
    parser.add_argument('--rgb2gray', type=bool, default=False, help="rgb to gray")
    args = parser.parse_args()

    generate_char_dataset = Generate_Char_Dataset(args)
    generate_char_dataset.run()