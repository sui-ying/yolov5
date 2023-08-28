import os
import cv2
import random
from PIL import Image
from PIL import ImageEnhance
from tqdm import tqdm
import numpy as np

image_path = '/media/westwell/60F64F35F64F0B2C/code/corner1/dataset/dataset_frontback/select/rgb/'
annot_path = '/media/westwell/60F64F35F64F0B2C/code/corner1/dataset/dataset_frontback/select/label/'
image_save_path = '/media/westwell/60F64F35F64F0B2C/code/corner1/dataset/dataset_frontback/select/enhance_rgb/'
annot_save_path = '/media/westwell/60F64F35F64F0B2C/code/corner1/dataset/dataset_frontback/select/enhence_label/'

if not os.path.exists(image_save_path):
    os.mkdir(image_save_path)
if not os.path.exists(annot_save_path):
    os.mkdir(annot_save_path)


# target_width = 1024
# target_height = 512

def enhance(image, annot):
    enhance_image = ImageEnhance.Brightness(image)
    bright = random.uniform(0.5, 1.5)
    enhance_image = enhance_image.enhance(bright)

    enhance_image = ImageEnhance.Contrast(enhance_image)
    contrast = random.uniform(0.5, 1.5)
    enhance_image = enhance_image.enhance(contrast)
    enhance_annot = annot

    #  if np.random.rand() > 0.5:
    #      enhance_image = enhance_image.transpose(Image.FLIP_LEFT_RIGHT)
    #      enhance_annot = enhance_annot.transpose(Image.FLIP_LEFT_RIGHT)

    #  if np.random.rand() > 0.5:
    #      enhance_image = enhance_image.transpose(Image.FLIP_TOP_BOTTOM)
    #      enhance_annot = enhance_annot.transpose(Image.FLIP_TOP_BOTTOM)
    return enhance_image, enhance_annot


counter = 0
for i in range(4):
    print("enhance : {} times".format(i))
    with tqdm(total=len(os.listdir(image_path))) as p_bar:
        for name in os.listdir(image_path):
            image = cv2.imread(image_path + name, 1)
            annot = cv2.imread(annot_path + name, 0)

            new_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            new_annot = Image.fromarray(annot)

            new_image, new_annot = enhance(new_image, new_annot)

            new_image.save(image_save_path + str(counter) + '_' + name)
            new_annot.save(annot_save_path + str(counter) + '_' + name)
        p_bar.update(1)
    counter += 1

