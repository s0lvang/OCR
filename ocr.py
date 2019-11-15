import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from skimage.filters import threshold_otsu
from skimage.feature import hog
from skimage import exposure
import math
import cv2
import matplotlib.pyplot as plt
import pickle


def img_round(x, base=75):
    return (base * math.floor(float(x)/base))


vround = np.vectorize(img_round)


def image_to_byte_array(path, size=40):
    img = cv2.imread(path)
    print(img[:, :, 0].shape)
    print(img.reshape(20, 20).shape)
    img = cv2.fastNlMeansDenoising(img).reshape(20, 20)
    binary = vround(img).flatten()
    print(binary.shape)
    print(get_image(path).shape)
    1/0
    return binary


def get_image(path, size=20):
    img = Image.open(path)
    img = img.convert("L")
    img = img.resize((size, size))
    return convert_image(img)


def invert_image(path):
    img = Image.open(path)
    img = img.convert("L")
    img = img.resize((20, 20))
    img = np.asarray(img)
    return ~img


def convert_image(img):
    image = np.asarray(img)
    if(not np.amin(image) == 255):
        thresh = threshold_otsu(image)
    else:
        thresh = [1 for i in range(len(image))]
    binary = image  # > thresh
    return get_hog(binary)


def get_hog(img):
    hog_image = hog(img, orientations=10, pixels_per_cell=(
        5, 5), cells_per_block=(3, 3))
    return exposure.rescale_intensity(hog_image, in_range=(0, 0.9))

def load_data():
    images = []
    dataset_path = "./dataset/chars74k-lite/"
    for char in os.listdir(dataset_path):
        images_on_char = [{'image': get_image(dataset_path + char + "/" + path), 'label': char} for path in os.listdir(
            dataset_path + char)]
        images_on_char += [{'image': convert_image(invert_image(dataset_path + char + "/" + path)), 'label': char} for path in os.listdir(
            dataset_path + char)]

        images += images_on_char
    return images


def check_white(window):
    count = np.count_nonzero(window == 255)
    return 1 - count/400


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1-x2)


def distance_to_rectangles(rectangles, xy):
    euclidean_distances = [euclidean_distance(
        np.array(rectangle), np.array(xy)) for rectangle in rectangles]
    return euclidean_distances if len(euclidean_distances) else [0]


def sliding_window(path, model):

    # read the image and define the stepSize and window size
    # (width,height)
    image = cv2.imread(path)
    tmp = image
    # for drawing a rectangle
    image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    stepSize = 15
    (w_width, w_height) = (20, 20)  # window size
    image_xy = []
    for x in range(0, image.shape[0] - w_width, stepSize):
        for y in range(0, image.shape[1] - w_height, stepSize):
            window = image[x:x + w_width, y:y + w_height]
            best_prediction = [[0]]
            best_white = 0
            label = ""
            white_score = check_white(window)
            if(white_score >= 0.3):
                positions = [i for i in range(-20, 20, 5)]
                for i in positions:
                    for j in positions:

                        window = image[x+j: x + j +
                                       w_width, y+i:y + i + w_height]
                        if(check_white(window) > best_white):
                            best_white = check_white(window)
                            if(window.shape == (20, 20)):
                                best_prediction = model.predict_proba(
                                    np.array(convert_image(window)).reshape(1, -1))
                            best_x, best_y = x+j, y+i

                # label = model.classes_[
                index = np.argmax(best_prediction)
                label = model.classes_[index]
                # print(label)
                cv2.putText(tmp, label,
                            (best_y+10, best_x - 20),
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=1,
                            color=(255, 0, 0))
                image_xy.append((best_x, best_y))
                cv2.rectangle(tmp, (best_y, best_x), (best_y + w_height, best_x + w_width),
                              (255, 0, 0), 2)  # draw rectangle on image
                plt.imshow(np.array(tmp).astype('uint8'))
    # show all windows
    plt.show()


def KNN():
    try:
        with open("knn.pkl", 'rb') as picklefile:
            model = pickle.load(picklefile)
    except:
        model = None
    if(not model):
        images = load_data()
        y = [image["label"]for image in images]
        x = [image["image"] for image in images]
        # inverted_x = [invert_image(image["image"])
        #              for image in images]
        #y += y
        #x += inverted_x
        X_train, X_test, Y_train, Y_test = train_test_split(x, y)
        model = KNeighborsClassifier(8)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        print(classification_report(Y_test, Y_pred))
        with open('knn.pkl', 'wb') as picklefile:
            pickle.dump(model, picklefile)
    sliding_window("./dataset/detection-images/detection-2.jpg", model)


def SVM():
    try:
        with open("SVM.pkl", 'rb') as picklefile:
            model = pickle.load(picklefile)
    except:
        model = None
    if(not model):
        images = load_data()
        y = [image["label"]for image in images]
        x = [image["image"] for image in images]
        # inverted_x = [invert_image(image["image"])
        #              for image in images]
        #y += y
        #x += inverted_x
        print(x)
        X_train, X_test, Y_train, Y_test = train_test_split(x, y)
        model = SVC(gamma="scale", probability=True)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        print(classification_report(Y_test, Y_pred))
        with open('SVM.pkl', 'wb') as picklefile:
            pickle.dump(model, picklefile)
    sliding_window("./dataset/detection-images/detection-2.jpg", model)


SVM()
