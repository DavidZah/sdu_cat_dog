# Created by David at 24.10.2022
# Project name sdu_cat_dog
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
        prediction = np.argmax(prediction, axis=-1)
        return prediction[0]

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
    path = "data\\catdog_data\\catdog_data\\validation"
    img_lst = get_file_list(path)
    labels = ["cat","dog"]
    predictor = Predictor(model_weigts="models\\save_at_5.h5")

    with Pool(12) as p:
        img_lst = p.map(load_img, img_lst)

    for i in img_lst:
        predicted = predictor.predict(i)
        i = cv2.cvtColor(i, cv2.COLOR_RGB2BGR)
        new_image = cv2.putText(
            img=i,
            text=labels[predicted],
            org=(0, 75),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=3.0,
            color=(128, 0, 0),
            thickness=3
        )
        cv2.imshow('new_image', i)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("done")