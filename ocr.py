import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from skimage.filters import threshold_otsu
import math
import cv2
import matplotlib.pyplot as plt


def img_round(x, base=75):
    return (base * math.floor(float(x)/base))


vround = np.vectorize(img_round)


def image_to_byte_array(path, size=40):
    img = Image.open(path)
    binary = vround(img).flatten()
    return binary


def load_data():
    images = []
    dataset_path = "./dataset/chars74k-lite/"
    for char in os.listdir(dataset_path):
        images_on_char = [{'image': image_to_byte_array(dataset_path + char + "/" + path), 'label': char} for path in os.listdir(
            dataset_path + char)]
        images += images_on_char
    return images




def main():
    images = load_data()
    y = [image["label"] for image in images]
    x = [image["image"] for image in images]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y)
    model = KNeighborsClassifier(7)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))


main()
