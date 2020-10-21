import csv
from sklearn.model_selection import train_test_split
import numpy
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import plot_model
from keras.models import Model
import os
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate


# define ensemble model

# fit stacked model on test dataset
datagen = ImageDataGenerator(
    rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
Xtrain = []
Y1 = []
Y2 = []
Y3 = []
Y4 = []
Y5 = []
with open('inputResnet.csv','r') as file:
    data = csv.reader(file)
    for row in data:
        print(row)
        x = []
        for j in range(0,20):

            x.append(float(row[j]))
        Xtrain.append(x)
        Y1.append(int(row[20]))
        Y2.append(int(row[21]))
        Y3.append(int(row[22]))
        Y4.append(int(row[23]))
        Y5.append(int(row[24]))


# print(Xtrain)
trainX = numpy.array(Xtrain)

visible = Input(shape=(20,))
c1 = Dense(16, activation='relu')(visible)
h1 = Dense(16, activation='relu')(c1)
h2 = Dense(32, activation='relu')(h1)
h3 = Dense(64, activation='relu')(h2)
h4 = Dense(128, activation='relu')(h3)
h5 = Dense(64, activation='relu')(h4)
h6 = Dense(32, activation='relu')(h5)
h7 = Dense(16, activation='relu')(h6)
c2 = Dense(16, activation='relu')(h3)
s1 = Dense(1,activation='sigmoid')(c2)
s2 = Dense(1,activation='sigmoid')(c2)
s3 = Dense(1,activation='sigmoid')(c2)
s4 = Dense(1,activation='sigmoid')(c2)
s5 = Dense(1,activation='sigmoid')(c2)

model = Model(inputs=visible,outputs=[s1,s2,s3,s4,s5])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# plot_model(model, show_shapes=True, to_file='ensemble.png')

model.fit(trainX,y= [Y1,Y2,Y3,Y4,Y5], epochs=100)
model.save("ensembleResnet1.h5")