import csv



# Xtrain = []
# with open('input.csv','r') as file:
#     data = csv.reader(file)
#     for row in data:
#         print(row)
#         x = []
#         y = []
#         for j in range(0,8):
#
#             x.append(float(row[j]))
#         Xtrain.append(x)
#         break
# print(clf.predict(Xtrain))


import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot
import matplotlib.pyplot as plt

model1 = load_model("resnet.h5")
model2 = load_model("resnet.h5")
model3 = load_model("resnet.h5")
model4 = load_model("resnet.h5")


modelx = load_model("ensembleResnet1.h5")
ytrue = []
ypred = []
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

Xtrain = []
Y1 = []
Y2 = []
Y3 = []
Y4 = []
Y5 = []
from os import walk
import shutil

f0 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\validation\\0'):
    f0.extend(filenames)
    break

f1 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\validation\\1'):
    f1.extend(filenames)
    break

f2 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\validation\\2'):
    f2.extend(filenames)
    break

f3 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\validation\\3'):
    f3.extend(filenames)
    break

f4 = []
for (dirpath, dirnames, filenames) in walk('E:\\diabeticRetinopathy\\validation\\4'):
    f4.extend(filenames)
    break
count = 0
total = 0
for i in f0:

    try:
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\0\\' + i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

    except:
        continue
    test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

    result1 = model1.predict_generator(test_generator)
    result2 = model2.predict_generator(test_generator)
    result3 = model3.predict_generator(test_generator)
    result4 = model4.predict_generator(test_generator)

    X = []
    X.extend(result1[0])
    X.extend(result2[0])
    X.extend(result3[0])
    X.extend(result4[0])


    Xtrain = np.array([X,])

    result = modelx.predict(Xtrain)


    xres = []
    print("hi")
    xsave = []
    for x in np.nditer(result):
        print(x)
        for j in range(0,len(x)):
            if x[j] < float(0.5):
                xres.append(0.0)
            else:
                xres.append(1.0)
        xsave = x
    print(xres)
    ytrue.append(0)
    if xres[0] == 1.0:
        count += 1
        print(count)
        ypred.append(0)
    else:
        m = -1
        for j in range(1, len(xres)):
            if xres[j] == 1:
                m = 0
                ypred.append(j)
                break
        if m == -1:
            max = -1
            ind = -1
            for j in range(0,len(xsave)):
                if xsave[j]>max:
                    max = xsave[j]
                    ind = j
            if ind == 0:
                count+=1
                print(count)
            ypred.append(ind)

    total += 1
    print(total)
    try:
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
    except:
        import time

        time.sleep(2)
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

for i in f1:

    try:
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\1\\' + i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

    except:
        continue
    test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

    result1 = model1.predict_generator(test_generator)
    result2 = model2.predict_generator(test_generator)
    result3 = model3.predict_generator(test_generator)
    result4 = model4.predict_generator(test_generator)

    X = []
    X.extend(result1[0])
    X.extend(result2[0])
    X.extend(result3[0])
    X.extend(result4[0])

    Xtrain = np.array([X, ])

    result = modelx.predict(Xtrain)


    xres = []
    print("hi")
    xsave = []
    for x in np.nditer(result):
        print(x)
        xsave = x
        for j in range(0,len(x)):
            if x[j] < float(0.5):
                xres.append(0.0)
            else:
                xres.append(1.0)
    print(xres)
    ytrue.append(1)
    if xres[0] == 1.0:

        ypred.append(0)
    else:
        m = -1
        for j in range(1, len(xres)):
            if xres[j] == 1:
                m = 0
                if j == 1:
                    count += 1
                    print(count)
                ypred.append(j)
                break
        if m == -1:
            max = -1
            ind = -1
            for j in range(0, len(xsave)):
                if xsave[j] > max:
                    max = xsave[j]
                    ind = j
            if ind == 1:
                count+=1
                print(count)
            ypred.append(ind)

    total += 1
    print(total)
    try:
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
    except:
        import time

        time.sleep(2)
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

for i in f2:

    try:
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\2\\' + i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

    except:
        continue
    test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

    result1 = model1.predict_generator(test_generator)
    result2 = model2.predict_generator(test_generator)
    result3 = model3.predict_generator(test_generator)
    result4 = model4.predict_generator(test_generator)
    X = []
    X.extend(result1[0])
    X.extend(result2[0])
    X.extend(result3[0])
    X.extend(result4[0])

    Xtrain = np.array([X, ])

    result = modelx.predict(Xtrain)

    xsave = []
    xres = []
    print("hi")
    for x in np.nditer(result):
        print(x)
        xsave = x
        for j in range(0,len(x)):
            if x[j] < float(0.5):
                xres.append(0.0)
            else:
                xres.append(1.0)
    print(xres)
    ytrue.append(2)
    if xres[0] == 1.0:

        ypred.append(0)
    else:
        m = -1
        for j in range(1, len(xres)):
            if xres[j] == 1:
                m = 0
                if j == 2:
                    count += 1
                    print(count)
                ypred.append(j)
                break
        if m == -1:
            max = -1
            ind = -1
            for j in range(0, len(xsave)):
                if xsave[j] > max:
                    max = xsave[j]
                    ind = j
            if ind == 2:
                count+=1
                print(count)
            ypred.append(ind)

    total += 1
    print(total)

    try:
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
    except:
        import time

        time.sleep(2)
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

for i in f3:

    try:
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\3\\' + i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

    except:
        continue
    test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

    result1 = model1.predict_generator(test_generator)
    result2 = model2.predict_generator(test_generator)
    result3 = model3.predict_generator(test_generator)
    result4 = model4.predict_generator(test_generator)
    X = []
    X.extend(result1[0])
    X.extend(result2[0])
    X.extend(result3[0])
    X.extend(result4[0])

    Xtrain = np.array([X,])

    result = modelx.predict(Xtrain)

    xsave = []
    xres = []
    print("hi")
    for x in np.nditer(result):
        print(x)
        xsave = x
        for j in range(0,len(x)):
            if x[j] < float(0.5):
                xres.append(0.0)
            else:
                xres.append(1.0)
    print(xres)
    ytrue.append(3)
    if xres[0] == 1.0:

        ypred.append(0)
    else:
        m = -1
        for j in range(1, len(xres)):
            if xres[j] == 1:
                m = 0
                if j == 3:
                    count += 1
                    print(count)
                ypred.append(j)
                break
        if m == -1:
            max = -1
            ind = -1
            for j in range(0, len(xsave)):
                if xsave[j] > max:
                    max = xsave[j]
                    ind = j
            if ind == 3:
                count+=1
                print(count)
            ypred.append(ind)

    total += 1
    print(total)

    try:
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
    except:
        import time

        time.sleep(2)
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

for i in f4:

    try:
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\4\\' + i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

    except:
        continue
    test_generator = datagen.flow_from_directory("E:\\diabeticRetinopathy\\gradcam_data\\")

    result1 = model1.predict_generator(test_generator)
    result2 = model2.predict_generator(test_generator)
    result3 = model3.predict_generator(test_generator)
    result4 = model4.predict_generator(test_generator)
    X = []
    X.extend(result1[0])
    X.extend(result2[0])
    X.extend(result3[0])
    X.extend(result4[0])

    Xtrain = np.array([X, ])

    result = modelx.predict(Xtrain)

    xsave = []
    xres = []
    print("hi")
    for x in np.nditer(result):
        print(x)
        xsave = x
        for j in range(0,len(x)):
            if x[j] < float(0.5):
                xres.append(0.0)
            else:
                xres.append(1.0)
    print(xres)
    ytrue.append(4)
    if xres[0] == 1.0:

        ypred.append(0)
    else:
        m = -1
        for j in range(1, len(xres)):
            if xres[j] == 1:
                m = 0
                if j == 4:
                    count += 1
                    print(count)
                ypred.append(j)
                break
        if m == -1:
            max = -1
            ind = -1
            for j in range(0, len(xsave)):
                if xsave[j] > max:
                    max = xsave[j]
                    ind = j
            ypred.append(ind)

    total += 1
    print(total)

    try:
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
    except:
        import time

        time.sleep(2)
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

mat = confusion_matrix(ytrue, ypred)
print(mat)
print(classification_report(ytrue, ypred))
scikitplot.metrics.plot_confusion_matrix(ytrue, ypred)
plt.show()
scikitplot.metrics.plot_confusion_matrix(ytrue, ypred, normalize=True)
plt.show()












