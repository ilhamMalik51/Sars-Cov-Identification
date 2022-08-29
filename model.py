import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import  LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
import MyUtils as mut
import Metrics as met

lokasi_dataset = "dataset"
(imdata, imlabel) = mut.load(lokasi_dataset)

## menormalisasikan seluruh dataset
X = imdata
X = X / 255

## mongconvert label
lb = LabelBinarizer()
## mengaplikasikan label
Ye = lb.fit_transform(imlabel)

## mensplit data sebesar 80% training data dan 15% test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Ye, random_state=42, shuffle=True, test_size=0.2, stratify=Ye)
print("Data sudah di-split")

## pembangunan model
model = Sequential()
model.add(Conv2D(64, (3, 3), strides=3, padding="same", activation="relu", 
                 input_shape=(192, 192, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))
model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))
model.add(Conv2D(128, (3, 3), strides=1, padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding="same"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=256, activation="relu"))
model.add(Dense(units=256, activation="relu"))
model.add(Dense(units=1, activation='sigmoid'))
print("Model berhasil dibuat!")

# 20 adam optimizer
# 20 10e-4
# 20 10e-5

print("Model berhasil dicompile")
opt = keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opt , loss="binary_crossentropy", metrics=["accuracy", met.f1_m, met.precision_m, met.recall_m])

print("Training dimulai")
model.fit(X_train, Y_train, epochs=20, verbose=1, validation_data=(X_test, Y_test))