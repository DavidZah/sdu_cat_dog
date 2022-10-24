# Created by David at 24.10.2022
# Project name sdu_cat_dog
import multiprocessing
import os
import random
from multiprocessing import Pool
from pathlib import Path

import cv2
import imutils
import random
from tqdm import tqdm
def augment_image_complete(img):
    img_dict = {}
    img_dict["original"] = img
    img_dict["flipVertical"] = cv2.flip(img, 0)
    img_dict["flipHorizontal"] = cv2.flip(img, 1)
    img_dict["flipBouth"] = cv2.flip(img, -1)

    img_dict["rotated_right"] = imutils.rotate_bound(img, random.randint(0, 85))
    img_dict["rotated_left"] = imutils.rotate_bound(img, random.randint(-175, -95))

    img_dict["rotated_right_half"] = imutils.rotate_bound(img, random.randint(95, 175))
    img_dict["rotated_left_half"] = imutils.rotate_bound(img, random.randint(-85, 5))

    img_dict["contrast_1"] = cv2.convertScaleAbs(img, alpha=random.randint(1,2), beta=random.randint(10,50))
    img_dict["contrast_2"] = cv2.convertScaleAbs(img, alpha=random.randint(2,3), beta=random.randint(50,90))

    img_dict["blurImg"] =cv2.blur(img, (7, 7))
    #img_dict["img_invert"] =cv2.bitwise_not(img)
    return img_dict

def load_augment_save(path,save_path,classe,index,dim =(512,512)):
    img = cv2.imread(path)
    img_dict = augment_image_complete(img)
    for z,key in enumerate(img_dict):
        img_resize = cv2.resize(img_dict[key], dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{save_path}/{classe}_{key}_{index}.png", img_resize)

def get_file_list(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            filelist.append(os.path.join(root, file))
    return filelist


def dogs_aug():
    file_lst = get_file_list("data/catdog_data/catdog_data/train/dogs")

    for i in range(len(file_lst)):
        load_augment_save(file_lst[i], "data/catdog_data/catdog_data/train_augmentation/dog", "dog", i)

def cats_aug():
    file_lst = get_file_list("data/catdog_data/catdog_data/train/cats")

    for i in range(len(file_lst)):
        load_augment_save(file_lst[i], "data/catdog_data/catdog_data/train_augmentation/cat", "cat", i)

if __name__ == "__main__":
    #image = cv2.imread("data/catdog_data/catdog_data/train/cats/cat.1.jpg")
    #load_augment_save("data/catdog_data/catdog_data/train/cats/cat.1.jpg","data/catdog_data/catdog_data/train_augmentation/cat","cat")

    p1 = multiprocessing.Process(target=dogs_aug)
    p = multiprocessing.Process(target=cats_aug)

    p1.start()
    p.start()

    p.join()
    p1.join()
    print("done")