from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import pickle
import os
import argparse

parser = argparse.ArgumentParser()
# Add optional and mandatory parameters
parser.add_argument("-t", "--train_set_location", help="Path to the test directory", default="./DataSet/Train Images/")
parser.add_argument("-m", "--model_file", help="Path and filename of the model file", default="./Model")
parser.add_argument("-c", "--sub_cat_file", help="Path and filename of the subcategories file", default="./Categories")
parser.add_argument("-e", "--input_epoch", help="Number of passes the code will make over the train set", default=5)
parser.add_argument("-b", "--batch_size", help="Batch size per epoch", default=32)
args = parser.parse_args()


# You can use any of the following pre-trained models to as weight
# 1. ImageNet
# 2. ResNet
# 3. VGG16 / VGG19

base_model = MobileNet(weights='imagenet',
                       include_top=False)  # imports the imagenet model and discards the last layer.

x = base_model.output
x = GlobalAveragePooling2D()(x)

# Adding one or more new layers provides more accurate results

x = Dense(1024, activation='relu')(x)  # dense layer 1
x = Dense(512, activation='relu')(x)  # dense layer 2
preds = Dense(2, activation='softmax')(x)  # final layer with softmax activation

model = Model(inputs=base_model.input, outputs=preds)

# specify the inputs & outputs
# now a model has been created based on our architecture
# Only train the new layers
for layer in model.layers[:2]:
    layer.trainable = False
for layer in model.layers[2:]:
    layer.trainable = True

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies

train_generator = train_datagen.flow_from_directory(args.train_set_location,  # Path to your training data. Each
                                                    # classification will be stored in a different directory
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=int(args.batch_size),
                                                    class_mode='categorical',
                                                    shuffle=True)

# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Choose an appropriate step size depending on the number of images present in your train set
step_size_train = train_generator.n // train_generator.batch_size

# Use trial and error to set the number of epochs that the model trains for. Each epoch represents one pass over the
# training set. Larger epochs will increase the chances of over-fitting the data, if you have a small training set
model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=int(args.input_epoch))

# Save this newly created model to a file so that it can be used later
file = open(args.model_file, 'wb')
pickle.dump(model, file=file)


dict = []
# Store all subcategories in a file as well
for filename in os.listdir(args.train_set_location):
    dict.append(filename)

cat_file = open(args.sub_cat_file, 'wb')
pickle.dump(dict, cat_file)
