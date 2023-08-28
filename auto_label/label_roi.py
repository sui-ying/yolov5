import os
import shutil

import cv2
import numpy as np

import xml.dom.minidom as md
import xml.etree.ElementTree as ET

import sys
sys.path.append("..")
from infer_img_roi import Infer_img_by_yolov5
sys.path.append("../../myscripts")
from tools import not_exists_path_make_dir, not_exists_path_make_dirs

def write_bbox_to_xml(xmlpath, filename, size, objects):
    """
    write cls & bbox info to VOC
    :param xmlpath: example.xml
    :param filename: example.jpg
    :param size: [640, 480, 3] # height, width, Channel
    :param objects: objects = [
                        {'name': 'person', 'xmin': 253, 'ymin': 123, 'xmax': 442, 'ymax': 367},
                        {'name': 'car', 'xmin': 100, 'ymin': 200, 'xmax': 300, 'ymax': 350}]
    :return:
    """

    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename

    size_element = ET.SubElement(root, "size")
    ET.SubElement(size_element, "height").text = str(size[0])
    ET.SubElement(size_element, "width").text = str(size[1])
    ET.SubElement(size_element, "depth").text = str(size[2])

    for obj in objects:
        object_element = ET.SubElement(root, "object")
        ET.SubElement(object_element, "name").text = obj['name']

        bndbox_element = ET.SubElement(object_element, "bndbox")
        ET.SubElement(bndbox_element, "xmin").text = str(obj['xmin'])
        ET.SubElement(bndbox_element, "ymin").text = str(obj['ymin'])
        ET.SubElement(bndbox_element, "xmax").text = str(obj['xmax'])
        ET.SubElement(bndbox_element, "ymax").text = str(obj['ymax'])

    xml_string = ET.tostring(root)  # 转换为字符串格式
    dom = md.parseString(xml_string)
    with open(xmlpath, 'w', encoding='utf-8') as f:
        f.write(dom.toprettyxml(indent='\t', encoding='utf-8').decode('utf-8'))


if __name__ == '__main__':
    # input
    input_path = "/home/westwell/Downloads/tmp/lorry/img"
    save_xml_path = "/home/westwell/Downloads/tmp/lorry/xml"
    error_path = os.path.join(input_path, "..", "error")
    yolo_no_roi_path = os.path.join(input_path, "..", "error/no_roi")
    not_exists_path_make_dirs([save_xml_path, error_path, yolo_no_roi_path])
    if not os.path.isdir(input_path):
        targets = [input_path]
    else:
        targets = [f for f in os.listdir(input_path) if not os.path.isdir(os.path.join(input_path, f))]
        targets = [os.path.join(input_path, f) for f in targets]

    infer_img_by_yolov7 = Infer_img_by_yolov5('/cv/best.pt')
    labelmap = infer_img_by_yolov7.labelmap
    print("labelmap: ", labelmap)

    for img_path in targets:
        imgfile = os.path.basename(img_path)
        imgname = os.path.splitext(imgfile)[0]
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        result_list = infer_img_by_yolov7.run(img)
        if len(result_list) == 0:
            print("yolov5 has not detect roi in image: ", imgfile)
            shutil.move(img_path, yolo_no_roi_path)
            continue

        try:
            # write bbox info to VOC(xml)
            objects = []
            for i in range(len(result_list)):
                info = result_list[i]
                _dict = {}
                _dict["name"] = labelmap[int(info[5])]
                _dict["xmin"] = int(info[0])
                _dict["ymin"] = int(info[1])
                _dict["xmax"] = int(info[2])
                _dict["ymax"] = int(info[3])
                objects.append(_dict)

            write_bbox_to_xml(os.path.join(save_xml_path, imgname+".xml"), imgfile, img.shape, objects)
        except Exception as E:
            print("{}, {}".format(imgname, E))
            shutil.move(img_path, error_path)
