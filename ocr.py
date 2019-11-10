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
import pickle


def img_round(x, base=75):
    return (base * math.floor(float(x)/base))


vround = np.vectorize(img_round)


def image_to_byte_array(path, size=40):
    img = cv2.imread(path)
    img = cv2.fastNlMeansDenoising(img)
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


def sliding_window(path, model):

    # read the image and define the stepSize and window size
    # (width,height)
    image = cv2.imread(path)
    tmp = image  # for drawing a rectangle
    stepSize = 5
    (w_width, w_height) = (20, 20)  # window size
    for x in range(0, image.shape[1] - w_width, stepSize):
        for y in range(0, image.shape[0] - w_height, stepSize):
            window = image[x:x + w_width, y:y + w_height, :]
            try:
                prediction = model.predict_proba(
                    np.array(vround(window).flatten()).reshape(1, -1))
            except:
                prediction = 0
        # classify content of the window with your classifier and
        # determine if the window includes an object (cell) or not
            # draw window on image
            if(np.amax(prediction) > 0.5):
                print(prediction)
                cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height),
                              (255, 0, 0), 2)  # draw rectangle on image
                plt.imshow(np.array(tmp).astype('uint8'))
    # show all windows
    plt.show()


def main():
    try:
        with open("knn.pkl", 'rb') as picklefile:
            model = pickle.load(picklefile)
    except:
        model = None
    if(not model):
        images = load_data()
        y = [image["label"] for image in images]
        x = [image["image"] for image in images]
        X_train, X_test, Y_train, Y_test = train_test_split(x, y)
        model = KNeighborsClassifier(10)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        print(classification_report(Y_test, Y_pred))
    with open('knn.pkl', 'wb') as picklefile:
        pickle.dump(model, picklefile)
    print("halla")
    sliding_window("./dataset/detection-images/detection-2.jpg", model)


main()
