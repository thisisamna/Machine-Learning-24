#!/usr/bin/env python
# coding: utf-8

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import os
import pandas as pd
import numpy as np
import cv2


path = '/home/amna/Documents/Assignment dataset'
categories = ['accordian','dollar_bill','motorbike','Soccer_Ball']

# Preprocessing train data
import imgaug.augmenters as iaa
contrast_sig = iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.4, 0.6))
vflip= iaa.Flipud(p=1.0) 

train_data = pd.DataFrame(columns=range(3780))
augmented_train_data = pd.DataFrame(columns=range(3780))



i=0
j=0
for category in categories:
    for img_name in os.listdir(path+'/train/'+category): 
        img = imread(path+'/train/'+category+'/'+img_name)
        resized_img=resize(img,(128,64))

        contrast_img = contrast_sig.augment_image(resized_img)   
        flipped_img= vflip.augment_image(resized_img)

        if(len(resized_img.shape)==3):
            fd = pd.Series(hog(resized_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=False, channel_axis=-1))
            contrast_fd = pd.Series(hog(contrast_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=False, channel_axis=-1))
            flipped_fd = pd.Series(hog(flipped_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=False, channel_axis=-1))

        else:
            fd = pd.Series(hog(resized_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=False, channel_axis=None))
            contrast_fd = pd.Series(hog(contrast_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=False, channel_axis=None))
            flipped_fd = pd.Series(hog(flipped_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=False, channel_axis=None))

        train_data.loc[i]=fd
        train_data.loc[i,"class"]=category
        i+=1
        
        augmented_train_data.loc[j]=fd
        augmented_train_data.loc[j,"class"]=category
        j+=1
        
        augmented_train_data.loc[j]=contrast_fd
        augmented_train_data.loc[j,"class"]=category
        j+=1
        
        augmented_train_data.loc[j]=flipped_fd
        augmented_train_data.loc[j,"class"]=category
        j+=1


train_data = train_data.sample(frac=1).reset_index(drop=True)
print(train_data.shape)


augmented_train_data=augmented_train_data.sample(frac=1).reset_index(drop=True)
print(augmented_train_data.shape)


y_train=train_data['class']
X_train=train_data.drop(columns='class')
y_train_aug=augmented_train_data['class']
X_train_aug=augmented_train_data.drop(columns='class')
print(X_train_aug.shape)
print(X_train.shape)
#Preprocessing test data
test_data = pd.DataFrame(columns=range(3780))
i=0
for category in categories:
    for img_name in os.listdir(path+'/test/'+category): 
        img = imread(path+'/test/'+category+'/'+img_name)
        resized_img=resize(img,(128,64))
        if(len(resized_img.shape)==3):
            fd = pd.Series(hog(resized_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=False, channel_axis=-1))
        else:
            fd = pd.Series(hog(resized_img,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2), visualize=False, channel_axis=None))
        test_data.loc[i]=fd
        test_data.loc[i,"class"]=category
        i+=1
y_test=test_data['class']
X_test=test_data.drop(columns='class')


# Function to train model and print metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
def evaluate_model(y, pred):
    print('Accuracy:')
    print(accuracy_score(y, pred))
    print('F1 Score:')
    print(f1_score(y,pred, average='weighted'))


from sklearn.svm import SVC
# Model 1 = original data
print('Model 2: augmented data')
svm = SVC()
svm.fit(X_train,y_train)
print('Training metrics')
pred_train=svm.predict(X_train)
evaluate_model(pred_train,y_train)
print('Testing metrics')
pred_test=svm.predict(X_test)
evaluate_model(pred_test,y_test)

print('==================================')

# Model 2 = augmented data
print('Model 2: augmented data')
svm = SVC()
svm.fit(X_train_aug,y_train_aug)
print('Training metrics')
pred_train_aug=svm.predict(X_train_aug)
evaluate_model(pred_train_aug,y_train_aug)
print('Testing metrics')
pred_test=svm.predict(X_test)
evaluate_model(pred_test,y_test)




