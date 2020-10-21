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






# load all models
model1=load_model("resnet.h5")
model2=load_model("resnet.h5")
model3=load_model("resnet.h5")
model4=load_model("resnet.h5")
from keras.models import Sequential

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
from os import walk
import shutil
f0 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\images_retina\\0'):

    f0.extend(filenames)
    break

f1 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\images_retina\\1'):

    f1.extend(filenames)
    break

f2 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\images_retina\\2'):

    f2.extend(filenames)
    break

f3 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\images_retina\\3'):

    f3.extend(filenames)
    break

f4 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\images_retina\\4'):
    f4.extend(filenames)
    break
count =0
with open('inputResnet.csv', mode='w') as file:
    filewriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    for i in f0:
        count+=1
        print(count)
        try:
            shutil.copyfile('E:\\diabeticRetinopathy\\images_retina\\0\\'+i,
                            "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

        except:
            continue
        test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

        result1 = model1.predict_generator(test_generator)
        result2 = model2.predict_generator(test_generator)
        result3 = model3.predict_generator(test_generator)
        result4 = model4.predict_generator(test_generator)
        x = []
        x.extend(result1[0])
        x.extend(result2[0])
        x.extend(result3[0])
        x.extend(result4[0])

        x.append(1)
        x.append(0)
        x.append(0)
        x.append(0)
        x.append(0)
        filewriter.writerow(x)

        try:
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
        except:
            import time

            time.sleep(2)
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

    for i in f1:
        count+=1
        print(count)
        try:
            shutil.copyfile('E:\\diabeticRetinopathy\\images_retina\\1\\'+i,
                            "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

        except:
            continue
        test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

        result1 = model1.predict_generator(test_generator)
        result2 = model2.predict_generator(test_generator)
        result3 = model3.predict_generator(test_generator)
        result4 = model4.predict_generator(test_generator)
        x = []
        x.extend(result1[0])
        x.extend(result2[0])
        x.extend(result3[0])
        x.extend(result4[0])

        x.append(0)
        x.append(1)
        x.append(0)
        x.append(0)
        x.append(0)
        filewriter.writerow(x)

        try:
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
        except:
            import time

            time.sleep(2)
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

    for i in f2:
        count+=1
        print(count)
        try:
            shutil.copyfile('E:\\diabeticRetinopathy\\images_retina\\2\\'+i,
                            "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

        except:
            continue
        test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

        result1 = model1.predict_generator(test_generator)
        result2 = model2.predict_generator(test_generator)
        result3 = model3.predict_generator(test_generator)
        result4 = model4.predict_generator(test_generator)
        x = []
        x.extend(result1[0])
        x.extend(result2[0])
        x.extend(result3[0])
        x.extend(result4[0])

        x.append(0)
        x.append(0)
        x.append(1)
        x.append(0)
        x.append(0)
        filewriter.writerow(x)

        try:
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
        except:
            import time

            time.sleep(2)
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)


    for i in f3:
        count+=1
        print(count)
        try:
            shutil.copyfile('E:\\diabeticRetinopathy\\images_retina\\3\\'+i,
                            "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

        except:
            continue
        test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

        result1 = model1.predict_generator(test_generator)
        result2 = model2.predict_generator(test_generator)
        result3 = model3.predict_generator(test_generator)
        result4 = model4.predict_generator(test_generator)
        x = []
        x.extend(result1[0])
        x.extend(result2[0])
        x.extend(result3[0])
        x.extend(result4[0])

        x.append(0)
        x.append(0)
        x.append(0)
        x.append(1)
        x.append(0)
        filewriter.writerow(x)

        try:
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
        except:
            import time

            time.sleep(2)
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

    for i in f4:
        count+=1
        print(count)
        try:
            shutil.copyfile('E:\\diabeticRetinopathy\\images_retina\\4\\'+i,
                            "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

        except:
            continue
        test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

        result1 = model1.predict_generator(test_generator)
        result2 = model2.predict_generator(test_generator)
        result3 = model3.predict_generator(test_generator)
        result4 = model4.predict_generator(test_generator)
        x = []
        x.extend(result1[0])
        x.extend(result2[0])
        x.extend(result3[0])
        x.extend(result4[0])

        x.append(0)
        x.append(0)
        x.append(0)
        x.append(0)
        x.append(1)
        filewriter.writerow(x)

        try:
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
        except:
            import time

            time.sleep(2)
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)