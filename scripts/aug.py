import cv2
import os
import sys
import  numpy as np
sys.path.append("/cv/xyc/myscripts")

from tqdm import tqdm
from tools import not_exists_path_make_dirs
from scripts_for_image import rotate
from convert import xywh_convert_xxyy, xxyy_convert_xywh

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']


def read_txt_and_return_list(txt_path):
    with open(txt_path) as fid:
        lines = fid.read().splitlines()
        while '' in lines:
            lines.remove('')
    return lines


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    shuchu = cv2.warpAffine(image, M, (nW, nH))
    # while (1):
    #     cv2.imshow('shuchu', shuchu)
    #     if cv2.waitKey(1) & 0xFF == 27:
    #         break
    return shuchu


def rotate180_img_label_and_save(oripath, savepath, visual_flag=False):

    img_path = os.path.join(oripath, "img")
    info_path = os.path.join(oripath, "info")

    save_img_path = os.path.join(savepath, "img")
    save_info_path = os.path.join(savepath, "info")
    save_visual_path = os.path.join(savepath, "visual")
    not_exists_path_make_dirs([save_img_path, save_info_path, save_visual_path])

    with tqdm(total=len(os.listdir(info_path))) as p_bar:
        p_bar.set_description("Rotate ...")
        for name in os.listdir(info_path):
            filename, _ = os.path.splitext(name)
            print(filename)
            imgfile = os.path.join(img_path, filename + ".jpg")
            infofile = os.path.join(info_path, name)

            img = cv2.imread(imgfile)
            H, W, C = img.shape
            # cv2.imshow("ori", img)
            # cv2.waitKey()

            rotate180_img = rotate_bound(img, 180)
            cv2.imwrite(os.path.join(save_img_path, filename + '.jpg'), rotate180_img)
            # cv2.imshow("rotate", rotate180_img)
            # cv2.waitKey()

            xml_info = read_txt_and_return_list(infofile)

            label_file = os.path.join(os.path.join(save_info_path, name + ".txt"))
            out_label_file = open(label_file, 'w')

            for i in range(len(xml_info)):
                cls, x, y, w, h = xml_info[i].split(" ")
                # xmin, ymin, xmax, ymax = xywh_convert_xxyy((W, H), (x, y, w, h))
                #
                # # after rotate: new xxyy
                # _xmax, _ymax = H-xmin, W-ymin
                # _xmin, _ymin = H-xmax, W-ymax
                #
                # x1, y1, w1, h1 = xxyy_convert_xywh((W, H), [_xmin, _ymin, _xmax, _ymax])
                # after rotate: new xxyy
                _x, _y = 1 - float(x), 1 - float(y)
                out_label_file.write(
                    str(cls) + " " + str(_x) + " " + str(_y) + " " + str(w) + " " + str(h) + "\n")

                # show img
                if visual_flag:
                    xmin, ymin, xmax, ymax = xywh_convert_xxyy((W, H), (_x, _y, w, h))
                    rotate180_img = cv2.rectangle(rotate180_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1)
                    rotate180_img = cv2.putText(rotate180_img, cls, (int(xmin), int(ymax)), cv2.FONT_HERSHEY_PLAIN, 1,
                                           (0, 255, 0), 1)
                    # cv2.imshow("R img", rotate180_img)
                    # cv2.waitKey()
                    cv2.imwrite(os.path.join(save_visual_path, name + ".jpg"), rotate180_img)
            out_label_file.close()

            p_bar.update(1)


# rotate180_img_label_and_save("/cv/xyc/Datasets/zongkong/det/rotate180",
#                              "/cv/xyc/Datasets/zongkong/det/rotate180_save", True)