from __future__ import print_function

import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix,classification_report
import scikitplot
import matplotlib.pyplot as plt
model=load_model("resnet.h5")
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator
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




import csv
count = 0
total = 0
y_true = []
y_pred = []
with open('benchmark.csv','r')as f, open('writeData.csv', mode='w') as file:
    data = csv.reader(f)
    filewriter = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    for row in data:

        try:
            shutil.copyfile("E:\\diabeticRetinopathy\\test_images\\"+row[0]+".png", "E:\\diabeticRetinopathy\\gradcam_data\\0\\"+row[0]+".png")

        except:
            continue
        test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")
        result = model.predict_generator(test_generator)

        i = np.argmax(result[0])
        print(i)
        print(row[1])
        if i == int(row[1]):
            print(count)
            count += 1
        y_true.append(int(row[1]))
        y_pred.append(i)
        x = []
        x.append(row[0])
        x.append(i)
        filewriter.writerow(x)
        total += 1
        print(total)
        try:
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + row[0] + ".png")
        except:
            import time
            time.sleep(2)
            os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + row[0] + ".png")
        if total==500:
            break
print(count)
print(total)
print(count/total*100)

mat = confusion_matrix(y_true, y_pred)
print(mat)
print(classification_report(y_true, y_pred))
scikitplot.metrics.plot_confusion_matrix(y_true,y_pred)
plt.show()
scikitplot.metrics.plot_confusion_matrix(y_true,y_pred,normalize=True)
plt.show()
