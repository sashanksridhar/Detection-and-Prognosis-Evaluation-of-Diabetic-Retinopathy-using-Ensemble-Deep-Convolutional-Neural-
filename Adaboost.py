import csv
Xtrain = []
Y1 = []
Y2 = []
Y3 = []
Y4 = []
Y5 = []
with open('input.csv','r') as file:
    data = csv.reader(file)
    for row in data:
        print(row)
        x = []
        y = []
        for j in range(0,8):

            x.append(float(row[j]))
        Xtrain.append(x)
        Y1.append(int(row[8]))
        Y2.append(int(row[9]))
        Y3.append(int(row[10]))
        Y4.append(int(row[11]))
        Y5.append(int(row[12]))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

clf1 = AdaBoostClassifier()
X_train, X_test, y_train, y_test = train_test_split(Xtrain, Y1, test_size=0.0, random_state=42)

clf1.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(Xtrain, Y2, test_size=0.0, random_state=42)
clf2 = AdaBoostClassifier()
clf2.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(Xtrain, Y3, test_size=0.0, random_state=42)
clf3 = AdaBoostClassifier()
clf3.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(Xtrain, Y4, test_size=0.0, random_state=42)
clf4 = AdaBoostClassifier()
clf4.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(Xtrain, Y5, test_size=0.0, random_state=42)
clf5 = AdaBoostClassifier()
clf5.fit(X_train,y_train)


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
from sklearn.metrics import confusion_matrix,classification_report
import scikitplot
import matplotlib.pyplot as plt
model1=load_model("class1.h5")
model2=load_model("class2.h5")
model3=load_model("class3.h5")
model4=load_model("class4.h5")

ytrue = []
ypred = []
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
count =0
total = 0
for i in f0:

    try:
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\0\\'+i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

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

    Xtrain = np.array([X])

    result1 = clf1.predict(Xtrain)
    result2 = clf2.predict(Xtrain)
    result3 = clf3.predict(Xtrain)
    result4 = clf4.predict(Xtrain)
    result5 = clf5.predict(Xtrain)

    xres = []
    print("hi")
    for x in np.nditer(result1):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result2):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result3):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result4):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result5):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    # print(xres)
    # print(row[1])
    # if xres[0] == 1.0:
    #     count += 1
    #     print(count)
    # ytrue.append(0)
    #
    # m = -1
    # for j in range(0,len(xres)):
    #     if xres[j]==1:
    #         m = 0
    #         ypred.append(j)
    #         break
    # if m == -1:
    #     ypred.append(0)
    ytrue.append(0)
    if xres[0] == 1.0:
        count+=1
        print(count)
        ypred.append(0)
    else:
        m=-1
        for j in range(1,len(xres)):
            if xres[j] == 1:
                 m = 0
                 count+=1
                 print(count)
                 ypred.append(1)
                 break
        if m == -1:
            ypred.append(0)

    total+=1
    print(total)
    try:
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)
    except:
        import time

        time.sleep(2)
        os.remove("E:\\diabeticRetinopathy\\gradcam_data\\0\\" + i)

for i in f1:

    try:
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\1\\'+i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

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

    Xtrain = np.array([X])

    result1 = clf1.predict(Xtrain)
    result2 = clf2.predict(Xtrain)
    result3 = clf3.predict(Xtrain)
    result4 = clf4.predict(Xtrain)
    result5 = clf5.predict(Xtrain)

    xres = []
    print("hi")
    for x in np.nditer(result1):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result2):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result3):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result4):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result5):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    # print(xres)
    # print(row[1])
    # if xres[1] == 1.0:
    #     count += 1
    #     print(count)
    ytrue.append(1)
    if xres[0] == 1.0:
        ypred.append(0)
    else:
        m = -1
        for j in range(1, len(xres)):
            if xres[j] == 1:
                m = 0
                count += 1
                print(count)
                ypred.append(1)
                break
        if m == -1:
            ypred.append(0)

    # m = -1
    # for j in range(0, len(xres)):
    #     if xres[j] == 1:
    #         m = 0
    #         ypred.append(j)
    #         break
    # if m == -1:
    #     ypred.append(0)
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
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\2\\'+i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

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

    Xtrain = np.array([X])

    result1 = clf1.predict(Xtrain)
    result2 = clf2.predict(Xtrain)
    result3 = clf3.predict(Xtrain)
    result4 = clf4.predict(Xtrain)
    result5 = clf5.predict(Xtrain)

    xres = []
    print("hi")
    for x in np.nditer(result1):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result2):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result3):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result4):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result5):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    # print(xres)
    # print(row[1])
    # if xres[2] == 1.0:
    #     count += 1
    #     print(count)
    # ytrue.append(2)
    #
    # m = -1
    # for j in range(0, len(xres)):
    #     if xres[j] == 1:
    #         m = 0
    #         ypred.append(j)
    #         break
    # if m == -1:
    #     ypred.append(0)
    ytrue.append(1)

    if xres[0] == 1.0:
        ypred.append(0)
    else:
        m = -1
        for j in range(1, len(xres)):
            if xres[j] == 1:
                m = 0
                count += 1
                print(count)
                ypred.append(1)
                break
        if m == -1:
            ypred.append(0)
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
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\3\\'+i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

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

    Xtrain = np.array([X])

    result1 = clf1.predict(Xtrain)
    result2 = clf2.predict(Xtrain)
    result3 = clf3.predict(Xtrain)
    result4 = clf4.predict(Xtrain)
    result5 = clf5.predict(Xtrain)

    xres = []
    print("hi")
    for x in np.nditer(result1):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result2):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result3):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result4):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result5):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    # print(xres)
    # print(row[1])
    # if xres[3] == 1.0:
    #     count += 1
    #     print(count)
    # ytrue.append(3)
    #
    # m = -1
    # for j in range(0, len(xres)):
    #     if xres[j] == 1:
    #         m = 0
    #         ypred.append(j)
    #         break
    # if m == -1:
    #     ypred.append(0)
    ytrue.append(1)

    if xres[0] == 1.0:
        ypred.append(0)
    else:
        m = -1
        for j in range(1, len(xres)):
            if xres[j] == 1:
                m = 0
                count += 1
                print(count)
                ypred.append(1)
                break
        if m == -1:
            ypred.append(0)
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
        shutil.copyfile('E:\\diabeticRetinopathy\\validation\\4\\'+i,
                        "E:\\diabeticRetinopathy\\gradcam_data\\0\\" +i)

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

    Xtrain = np.array([X])

    result1 = clf1.predict(Xtrain)
    result2 = clf2.predict(Xtrain)
    result3 = clf3.predict(Xtrain)
    result4 = clf4.predict(Xtrain)
    result5 = clf5.predict(Xtrain)

    xres = []
    print("hi")
    for x in np.nditer(result1):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result2):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result3):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result4):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    for x in np.nditer(result5):

        if x < float(0.5):
            xres.append(0.0)
        else:
            xres.append(1.0)
    # print(xres)
    # print(row[1])
    # if xres[4] == 1.0:
    #     count += 1
    #     print(count)
    # ytrue.append(4)
    #
    # m = -1
    # for j in range(0, len(xres)):
    #     if xres[j] == 1:
    #         m = 0
    #         ypred.append(j)
    #         break
    # if m == -1:
    #     ypred.append(0)
    ytrue.append(1)
    if xres[0] == 1.0:
        ypred.append(0)
    else:
        m = -1
        for j in range(1, len(xres)):
            if xres[j] == 1:
                m = 0
                count += 1
                print(count)
                ypred.append(1)
                break
        if m == -1:
            ypred.append(0)
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
scikitplot.metrics.plot_confusion_matrix(ytrue,ypred)
plt.show()
scikitplot.metrics.plot_confusion_matrix(ytrue,ypred,normalize=True)
plt.show()












