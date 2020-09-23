from google.colab import drive
drive.mount('/content/drive',force_remount=True)
import numpy as np 
import pandas as pd
from PIL import Image #to open image
import os
from os import listdir
from pathlib import Path
import cv2
import random
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
pd.options.mode.chained_assignment = None 
demo_neg =Image.open('/content/drive/My Drive/BrainTumor/no/38 no.jpg').resize((128,128))
demo_neg
demo_pos =Image.open('/content/drive/My Drive/BrainTumor/yes/Y10.jpg').resize((128,128))
demo_pos
directory = "/content/drive/My Drive/BrainTumor"
type = ['yes', 'no']
img_data=[]
data_label =[]
data_set =[]
for image_type in type:
    path = os.path.join(directory, image_type)
    img_type_index = type.index(image_type)
    for image in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
        img_array = img_array.astype(np.float32)
        img_array = cv2.resize(img_array, (128, 128))
        img_data.append([img_array, img_type_index])
random.shuffle(img_data)
for array_img, data_type in img_data:
    data_set.append(array_img)
    data_label.append(data_type)
len(data_label)
x_train, x_test, y_train, y_test = train_test_split(data_set, data_label, test_size = 0.2, 
                                                    random_state = 45)
x_train = np.array(x_train).reshape(-1,128,128, 1)
x_train = x_train/255.0
x_test = np.array(x_test).reshape(-1,128,128, 1)
y_train = np.array(y_train)
model = Sequential()
model.add(Conv2D( 128,(3,3), padding='same', input_shape = x_train.shape[1:], activation = "relu"))
model.add(Conv2D(128, (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
model.add(Dense(512))
model.add(Dense(1,activation='sigmoid'))
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model = Sequential()
model.add(Conv2D( 128,(3,3), padding='same', input_shape = x_train.shape[1:], activation = "relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(Dense(1,activation='sigmoid'))
model.summary()
# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
#Train The model
x = model.fit(
    x_train,
    y_train,
    batch_size=50,
    epochs=20,
    validation_split = 0.1
)
# Save neural network structure
model_structure = model.to_json()
f = Path('/content/drive/My Drive/model/model _structure.json')
f.write_text(model_structure)
# Save neural network's trained weights
model.save_weights("/content/drive/My Drive/model/model_weights.h5")
from IPython.display import SVG
from IPython.core.pylabtools import figsize
figsize = (1,1)
import pydot
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model,show_shapes = True,dpi=60).create(prog='dot', format='svg'))


