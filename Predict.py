from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import numpy as np
from collections import OrderedDict
import pickle
import argparse
import os

parser = argparse.ArgumentParser()
# Add optional and mandatory parameters
parser.add_argument("-t", "--test_set_location", help="Path to the test directory", default="./DataSet/Test Images/")
parser.add_argument("-m", "--model_file", help="Path and file name of model file", default="./Model")
parser.add_argument("-c", "--subcategories", help="Path and name of sub categories file", default="./Categories")
args = parser.parse_args()

# Load the model file to predict images
filename = open(args.model_file, 'rb')
model = pickle.load(filename)

# Load all the subclasses
categories = open(args.subcategories, 'rb')
category_dict = pickle.load(categories)

# Load the image to be predicted
# You can put the below code in a loop to predict multiple images in one run
for filename in os.listdir(args.test_set_location):
    img_path = args.test_set_location + filename
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    prediction = model.predict(x)
    index = 0
    maximum_value = prediction[0][index]
    i = 0
    for temp in prediction:
        for pred in temp:
            if pred > maximum_value:
                maximum_value = pred
                index = i
            i += 1
        print(filename + " : " + category_dict[index])
