# Customized DL Model for Alzheimers Detection from MRI Images

**This Project Aims to Develop a Customized Deep Learning Model which Detects 4 Classes of Alzheimer (Non Demented,
Very Mild Demented, Mild Demented and Moderate Demented) from MRI Images. The goal is to improve early detection and
diagnosis of Alzheimer’s disease using non-invasive and accessible imaging technology**

## Models and Files from this project
```
- Model Model_1
- Model Model_1.h5
- Model Model_2
- Model Model_2.h5
- Model Model_Inception
- Model Model_Inception.h5
- Variables.npz
```
<hr>

## Libraries and Data
Goal is to Train a Deep Learning Model on Image Data type. The trained model will be able to detect different classes of alzheimer from MRI Images. 
```python
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from distutils.dir_util import copy_tree, remove_tree
from PIL import Image
from random import randint
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import balanced_accuracy_score as BAS
from sklearn.metrics import classification_report, confusion_matrix 
import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPool2D
import warnings
warnings.filterwarnings('ignore')
print("TensorFlow Version:", tf.__version__)
```
**Configurations**
```python
base_dir = "Alzheimer_s Dataset/"
root_dir = "./"
test_dir = base_dir + "test/"
train_dir = base_dir + "train/"
work_dir = root_dir + "dataset/"
if os.path.exists(work_dir):
    remove_tree(work_dir)
os.mkdir(work_dir)
copy_tree(train_dir, work_dir)
copy_tree(test_dir, work_dir)
print("Working Directory Contents:", os.listdir(work_dir))

WORK_DIR = work_dir

CLASSES = [ 'NonDemented',
            'VeryMildDemented',
            'MildDemented',
            'ModerateDemented']

IMG_SIZE = 176
IMAGE_SIZE = [176, 176]
DIM = (IMG_SIZE, IMG_SIZE)
```
**Data Augmentation**
```python
ZOOM = [0.9, 1.1]
BRIGHT_RANGE = [0.8, 1.2]
HORZ_FLIP = True
FILL_MODE = "constant"
DATA_FORMAT = "channels_last"

work_dr = IDG(rescale = 1./255, brightness_range=BRIGHT_RANGE, zoom_range=ZOOM, data_format=DATA_FORMAT, fill_mode=FILL_MODE, horizontal_flip=HORZ_FLIP)

train_data_gen = work_dr.flow_from_directory(directory=WORK_DIR, target_size=DIM, batch_size=7000, shuffle=True)
```
```python
def show_images(generator,y_pred=None):
    
    # get image lables
    labels =dict(zip([0,1,2,3], CLASSES))
    
    # get a batch of images
    x,y = generator.next()
    
    # display a grid of 9 images
    plt.figure(figsize=(10, 10))
    if y_pred is None:
        for i in range(12):
            ax = plt.subplot(4, 3, i + 1)
            idx = randint(0, 6400)
            plt.imshow(x[idx])
            plt.axis("off")
            plt.title("Class:{}".format(labels[np.argmax(y[idx])]))
                                                     
    else:
        for i in range(12):
            ax = plt.subplot(4, 3, i + 1)
            plt.imshow(x[i])
            plt.axis("off")
            plt.title("Actual:{} \nPredicted:{}".format(labels[np.argmax(y[i])],labels[y_pred[i]]))
# Display Train Images
show_images(train_data_gen)
```

![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/7d1521eb-1394-4b03-87de-39a7c64c5cda)

<hr>

## Data Processing - Class Imbalance
```python
#Retrieving the data from the ImageDataGenerator iterator
train_data_x, train_labels_y = train_data_gen.next()
```
**Distribution of Classes**
```python
def plot_dist(train_data, train_labels):
    labels =dict(zip([0,1,2,3], CLASSES))

    dict_images = {c:0 for c in CLASSES}
    
    for i in range(0,train_data.shape[0]):
        dict_images[labels[np.argmax(train_labels[i])]] += 1
    
    print(dict_images)
        
    sorted_dict_images = sorted(dict_images.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_dict_images)

    colors = np.linspace(0.2, 0.8, len(classes))

    plt.bar(range(len(classes)), counts, color=plt.cm.Reds(colors), width=0.5)

    for i, (class_label, count) in enumerate(sorted_dict_images):
        plt.text(i, count+50, count, ha='center')

    plt.xticks(range(len(classes)), classes)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Classes')

    plt.show()
```
![image](https://github.com/user-attachments/assets/373d5a60-e45b-4448-b18e-e1347d649cb4)

### Applying SMOTE to solve the problem of Class Imnbalance
The Synthetic Minority Over-sampling Technique (SMOTE) is a popular method used to address the issue of class imbalance in datasets, particularly in the context of machine learning. Class imbalance occurs when the number of instances in one class significantly outnumbers those in another, which can lead to biased models that perform poorly on the minority class. SMOTE tackles this problem by generating synthetic examples of the minority class. It does this by selecting two or more similar instances from the minority class and creating new instances that are interpolations of these selected instances. This approach helps to balance the class distribution without simply duplicating existing minority instances, which can lead to overfitting.

SMOTE is particularly useful in scenarios where the minority class is underrepresented, such as fraud detection, medical diagnosis, and rare event prediction. By creating synthetic samples, SMOTE enhances the model's ability to learn the characteristics of the minority class, leading to better generalization and improved performance on unseen data. However, it is important to note that SMOTE can introduce noise if not used carefully, as the synthetic samples are not real data points but interpolations. Therefore, it is often used in conjunction with other techniques, such as Tomek links or Edited Nearest Neighbors, to refine the dataset and ensure the quality of the synthetic samples.
```python
sm = SMOTE(random_state=42)
train_data_x, train_labels_y = sm.fit_resample(train_data_x.reshape(-1, IMG_SIZE * IMG_SIZE * 3), train_labels_y)
train_data_x = train_data_x.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
print(train_data_x.shape, train_labels_y.shape)
plot_dist(train_data_x, train_labels_y)
```
![image](https://github.com/user-attachments/assets/1dc14b63-4cd3-4a38-b1b3-c845fbe4d0a7)

### Data Splitting into Training, Validation and Testing Data
```python
train_data, test_data, train_labels, test_labels = train_test_split(train_data_x, train_labels_y, test_size = 0.22, random_state=42)
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.22, random_state=42)
print(f'Training Size : {train_data.shape} | Training Labels : {train_labels.shape}')
print(f'Validation Size : {val_data.shape} | Validation Labels : {val_labels.shape}')
print(f'Testing Size : {test_data.shape}   |  Testing Labels : {test_labels.shape}')
```
![image](https://github.com/user-attachments/assets/5774ea30-04a6-429d-bf3d-262621199ac6)

<hr>

## Constructing Model


## Model Model_1
```
- Only Custom Callbacks
- Dropout after conv_block(64)
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/72e3be04-e150-436f-91bc-54e751b6fc43)
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/2b37330c-ad24-4a16-9cdb-5eb18c0cf881)

### Results from Model_1
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/077eddb4-a0f7-4d9b-916a-be642094137c)
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/483dfdc2-3043-4124-8e6c-b067ae87fc34)

<hr>

## Model Model_2
```
- Custom + Reduce LR on Plateau Callbacks
- No Droput After conv block(64)
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/566fade7-7b9c-4ab7-ae39-2f1bf6d7a365)
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/0445f7fc-f67a-4b81-a419-400f160ab375)

### Results from Model_2
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/6faff82e-a00a-4e17-9348-e98e9a480803)
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/0c874f05-09e0-43a4-a352-bb451eaaf26b)

<hr>

## Model Inception V3
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/ef8b9a85-e9f3-4c82-bbe8-2cb90ccb7601)
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/a4b0beb5-cfd9-4b9f-91fb-0c8ef1be8cd4)
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/a6d1201f-add4-46b5-9e68-1ce56b50ec48)

### Results from Inception V3
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/2558ed8c-bc2f-480d-9262-57467e5bc051)
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/16a9294e-7535-49ff-b661-f8fa26eb2cbb)




