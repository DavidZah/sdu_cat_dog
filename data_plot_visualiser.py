# Created by David at 01.11.2022
# Project name sdu_cat_dog
# Created by David at 24.10.2022
# Project name sdu_cat_dog
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Predictor:
    def __init__(self,model = None,model_weigts = None):
        if model == None:
            model = self.make_model()

        if model_weigts == None and model == None:
            raise AttributeError

        model.load_weights(model_weigts)
        self.model = model

    def make_model(self,input_shape=(512, 512,3)):
        model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            alpha=1.0,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None,
            classes=2,
            classifier_activation="softmax",
        )
        return model

    def predict(self,img):
        data = np.expand_dims(img, axis=0)
        prediction = self.model.predict(data)
        return prediction

def get_file_list(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            # append the file name to the list
            filelist.append(os.path.join(root, file))
    return filelist

def load_img(path,dim =(512,512)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img_resize



if __name__ =="__main__":
    path_cat = "data\\catdog_data\\catdog_data\\test\\cats"
    path_dog = "data\\catdog_data\\catdog_data\\test\\dogs"
    img_lst_cat = get_file_list(path_cat)
    img_lst_dog = get_file_list(path_dog)
    labels = ["cat","dog"]
    predictor = Predictor(model_weigts="models\\save_at_5_0.9075.h5")

    with Pool(12) as p:
        img_lst_cat = p.map(load_img, img_lst_cat)

    with Pool(12) as p:
        img_lst_dog = p.map(load_img, img_lst_dog)

    data_lst_cat = []
    data_lst_dog = []
    for i in img_lst_cat:
        predicted = predictor.predict(i)


        data_lst_cat.append(predicted.tolist()[0])

    for i in img_lst_dog:
        predicted = predictor.predict(i)

        data_lst_dog.append(predicted.tolist()[0])

    x,y = zip(*data_lst_cat)
    b,c = zip(*data_lst_dog)

    plt.scatter(x,y,"ro")
    plt.scatter(b,c, "bo")
    plt.show()

    print("done")