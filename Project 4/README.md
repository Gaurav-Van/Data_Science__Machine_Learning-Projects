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

Creating a Convolutional Neural Network (CNN) model using generic functions and classes to handle different parameters. The model is designed 
for image classification tasks and includes custom callback functionality to stop training once a desired accuracy is achieved.

### Functions and Classes

1. **Convolutional Block Function (`conv_block`)**
   - **Purpose**: Creates a block of convolutional layers with specified filters and activation functions.
   - **Parameters**:
     - `filters`: Number of filters for the convolutional layers.
     - `act`: Activation function to use (default is `'relu'`).
   - **Returns**: A `Sequential` model containing two convolutional layers, batch normalization, and max pooling.

   ```python
   def conv_block(filters, act='relu'):
       block = Sequential()
       block.add(Conv2D(filters, 3, activation=act, padding='same'))
       block.add(Conv2D(filters, 3, activation=act, padding='same'))
       block.add(BatchNormalization())
       block.add(MaxPool2D())
       return block
   ```

2. **Dense Block Function (`dense_block`)**
   - **Purpose**: Creates a block of dense layers with specified units, dropout rate, and activation functions.
   - **Parameters**:
     - `units`: Number of units for the dense layers.
     - `dropout_rate`: Dropout rate for regularization.
     - `act`: Activation function to use (default is `'relu'`).
   - **Returns**: A `Sequential` model containing two dense layers, batch normalization, and dropout.

   ```python
   def dense_block(units, dropout_rate, act='relu'):
       block = Sequential()
       block.add(Dense(units, activation=act))
       block.add(Dense(units, activation=act))
       block.add(BatchNormalization())
       block.add(Dropout(dropout_rate))
       return block
   ```

3. **Model Construction Function (`construct_model`)**
   - **Purpose**: Constructs the CNN model using the defined convolutional and dense blocks.
   - **Parameters**:
     - `act`: Activation function to use throughout the model (default is `'relu'`).
   - **Returns**: A `Sequential` model ready for training.

   ```python
   def construct_model(act='relu'):
       model = Sequential([
           Input(shape=(*IMAGE_SIZE, 3)),
           Conv2D(16, 3, activation=act, padding='same'),
           Conv2D(16, 3, activation=act, padding='same'),
           MaxPool2D(),
           conv_block(32),
           conv_block(64),
           Dropout(0.2),
           conv_block(128),
           Dropout(0.2),
           conv_block(256),
           Dropout(0.2),
           Flatten(),
           dense_block(512, 0.7),
           dense_block(128, 0.5),
           dense_block(64, 0.3),
           Dense(4, activation='softmax')
       ], name="cnn_model")
       return model
   ```

4. **Custom Callback Class (`MyCallback`)**
   - **Purpose**: Custom callback to stop training when validation accuracy exceeds 99%.
   - **Methods**:
     - `on_epoch_end`: Checks validation accuracy at the end of each epoch and stops training if the threshold is met.

   ```python
   class MyCallback(tf.keras.callbacks.Callback):
       def on_epoch_end(self, epoch, logs={}):
           if logs.get('val_acc') > 0.99:
               print("\nReached accuracy threshold! Terminating training.")
               self.model.stop_training = True
   ```

#### Model Compilation and Summary
- **Callbacks**: List of callbacks to use during training, including the custom `MyCallback`.
- **Metrics**: List of metrics to evaluate the model, including categorical accuracy, AUC, and F1 score.
- **Compilation**: The model is compiled with the Adam optimizer, categorical cross-entropy loss, and specified metrics.

```python
my_callback = MyCallback()
model = construct_model()
CALLBACKS = [my_callback]
METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
           tf.keras.metrics.AUC(name='auc'), 
           tfa.metrics.F1Score(num_classes=4)]
model.compile(optimizer='adam',
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=METRICS)
model.summary()
```
![image](https://github.com/user-attachments/assets/14fc4130-ef83-4182-a578-3fdf007ca38e)

```python
EPOCHS = 100
history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), callbacks=CALLBACKS, epochs=EPOCHS)
```

<hr>

## Model Model_1

**Parameter and Configuration**
```
- Only Custom Callbacks
- Dropout after conv_block(64)
```
```python
model_dir = "./Model " + "Model_1"
model.save(model_dir, save_format='h5')
model1 = tf.keras.models.load_model("/kaggle/working/Model Model_1")
model1.summary()
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/72e3be04-e150-436f-91bc-54e751b6fc43)

```python
fig, ax = plt.subplots(1, 3, figsize = (30, 5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "auc", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/2b37330c-ad24-4a16-9cdb-5eb18c0cf881)

### Results from Model_1
```python
train_scores = model1.evaluate(train_data, train_labels)
val_scores = model1.evaluate(val_data, val_labels)
test_scores = model1.evaluate(test_data, test_labels)
print("Training Accuracy: %.2f%%"%(train_scores[1] * 100))
print("Validation Accuracy: %.2f%%"%(val_scores[1] * 100))
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/077eddb4-a0f7-4d9b-916a-be642094137c)

**Classification Report**
```python
CLASSES = [ 'NonDemented',
            'VeryMildDemented',
            'MildDemented',
            'ModerateDemented']

pred_labels = model1.predict(test_data)

def roundoff(arr):
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in pred_labels:
    labels = roundoff(labels)

print(classification_report(test_labels, pred_labels, target_names=CLASSES))
```
![image](https://github.com/user-attachments/assets/a0f3031d-3f26-4604-99da-41da3e72d4a0)

**Balanced Accuracy Score and Matthew's Correlation Coefficient**

**Balanced Accuracy** Score is a metric used to evaluate the performance of a classification model, especially in the presence of imbalanced datasets. It is defined as the average of sensitivity (true positive rate) and specificity (true negative rate). This metric is particularly useful when the classes are imbalanced, as it gives equal weight to both classes, ensuring that the performance on the minority class is not overshadowed by the majority class.

**Matthew’s Correlation Coefficient (MCC)** is another robust metric for evaluating the performance of binary classification models. It takes into account all four quadrants of the confusion matrix: true positives, true negatives, false positives, and false negatives. MCC is particularly valued for its ability to provide a balanced measure even when the classes are imbalanced. The coefficient ranges from -1 to +1, where +1 indicates perfect prediction, 0 indicates no better than random prediction, and -1 indicates total disagreement between prediction and observation. 

```python
print("Balanced Accuracy Score: {} %".format(round(BAS(test_ls, pred_ls) * 100, 2)))
print("Matthew's Correlation Coefficient: {} %".format(round(MCC(test_ls, pred_ls) * 100, 2)))
```
![image](https://github.com/user-attachments/assets/e4f2b3b8-b5a5-45a9-ac98-f4f69cc59d1d)

![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/483dfdc2-3043-4124-8e6c-b067ae87fc34)

<hr>

## Model Model_2

**Parameter and Configuration**
```
- Custom + Reduce LR on Plateau Callbacks
- No Droput After conv block(64)
```
```python
model_dir = "./Model " + "Model_2"
model.save(model_dir, save_format='h5')
model2 = tf.keras.models.load_model("/kaggle/working/Model Model_2")
model2.summary()
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/566fade7-7b9c-4ab7-ae39-2f1bf6d7a365)
```python
fig, ax = plt.subplots(1, 3, figsize = (30, 5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "auc", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/0445f7fc-f67a-4b81-a419-400f160ab375)

### Results from Model_2
```python
train_scores = model2.evaluate(train_data, train_labels)
val_scores = model2.evaluate(val_data, val_labels)
test_scores = model2.evaluate(test_data, test_labels)
print("Training Accuracy: %.2f%%"%(train_scores[1] * 100))
print("Validation Accuracy: %.2f%%"%(val_scores[1] * 100))
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/6faff82e-a00a-4e17-9348-e98e9a480803)

**Classification Report**
```python
CLASSES = [ 'NonDemented',
            'VeryMildDemented',
            'MildDemented',
            'ModerateDemented']

pred_labels = model2.predict(test_data)

def roundoff(arr):
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in pred_labels:
    labels = roundoff(labels)

print(classification_report(test_labels, pred_labels, target_names=CLASSES))
```
![image](https://github.com/user-attachments/assets/63c4e6a0-a4bc-48cc-8580-183a27df5500)

**Balanced Accuracy Score and Matthew's Correlation Coefficient**

**Balanced Accuracy** Score is a metric used to evaluate the performance of a classification model, especially in the presence of imbalanced datasets. It is defined as the average of sensitivity (true positive rate) and specificity (true negative rate). This metric is particularly useful when the classes are imbalanced, as it gives equal weight to both classes, ensuring that the performance on the minority class is not overshadowed by the majority class.

**Matthew’s Correlation Coefficient (MCC)** is another robust metric for evaluating the performance of binary classification models. It takes into account all four quadrants of the confusion matrix: true positives, true negatives, false positives, and false negatives. MCC is particularly valued for its ability to provide a balanced measure even when the classes are imbalanced. The coefficient ranges from -1 to +1, where +1 indicates perfect prediction, 0 indicates no better than random prediction, and -1 indicates total disagreement between prediction and observation. 

```python
print("Balanced Accuracy Score: {} %".format(round(BAS(test_ls, pred_ls) * 100, 2)))
print("Matthew's Correlation Coefficient: {} %".format(round(MCC(test_ls, pred_ls) * 100, 2)))
```
![image](https://github.com/user-attachments/assets/85213e63-2f06-4a82-b896-4ad58b82a06d)

![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/0c874f05-09e0-43a4-a352-bb451eaaf26b)

<hr>

## Model Inception V3

InceptionV3 is a deep convolutional neural network architecture that has been pre-trained on the ImageNet dataset. It is known for its efficiency and high performance in image classification tasks. The architecture includes multiple convolutional layers and inception modules that capture various levels of abstraction in the input images. By leveraging pre-trained weights, InceptionV3 can be fine-tuned for specific tasks with relatively small datasets.

1. **Loading the Pre-trained InceptionV3 Model**
   - **Purpose**: Initialize the InceptionV3 model with pre-trained weights from ImageNet, excluding the top fully connected layers.
   - **Parameters**:
     - `input_shape`: Specifies the shape of the input images.
     - `include_top`: Set to `False` to exclude the fully connected layers at the top of the network.
     - `weights`: Specifies the pre-trained weights to load.

   ```python
   inception_model = InceptionV3(input_shape=(176, 176, 3), include_top=False, weights="imagenet")
   ```

2. **Freezing the Layers of InceptionV3**
   - **Purpose**: Prevent the weights of the pre-trained InceptionV3 layers from being updated during training.
   - **Implementation**: Iterate over each layer in the InceptionV3 model and set `trainable` to `False`.

   ```python
   for layer in inception_model.layers:
       layer.trainable = False
   ```

3. **Custom Model Construction**
   - **Purpose**: Build a custom sequential model using the pre-trained InceptionV3 as the base, followed by additional layers for further processing and classification.
   - **Layers**:
     - `Dropout`: Regularization layer to prevent overfitting.
     - `GlobalAveragePooling2D`: Reduces each feature map to a single value, maintaining spatial information.
     - `Flatten`: Flattens the input to a 1D array.
     - `BatchNormalization`: Normalizes the activations of the previous layer.
     - `Dense`: Fully connected layers with specified units and activation functions.

   ```python
   custom_inception_model = Sequential([
       inception_model,
       Dropout(0.5),
       GlobalAveragePooling2D(),
       Flatten(),
       BatchNormalization(),
       Dense(512, activation='relu'),
       BatchNormalization(),
       Dropout(0.5),
       Dense(256, activation='relu'),
       BatchNormalization(),
       Dropout(0.5),
       Dense(128, activation='relu'),
       BatchNormalization(),
       Dropout(0.5),
       Dense(64, activation='relu'),
       Dropout(0.5),
       BatchNormalization(),
       Dense(4, activation='softmax')
   ], name="inception_cnn_model")
   ```

4. **Custom Callback Class**
   - **Purpose**: Define a custom callback to stop training when the accuracy exceeds 99%.
   - **Method**:
     - `on_epoch_end`: Checks the accuracy at the end of each epoch and stops training if the threshold is met.

   ```python
   class MyCallback(tf.keras.callbacks.Callback):
       def on_epoch_end(self, epoch, logs={}):
           if logs.get('acc') > 0.99:
               print("\nReached accuracy threshold! Terminating training.")
               self.model.stop_training = True
   my_callback = MyCallback()
   ```

5. **ReduceLROnPlateau Callback**
   - **Purpose**: Reduce the learning rate when the validation loss plateaus to stabilize the training process.
   - **Parameters**:
     - `monitor`: Metric to be monitored.
     - `patience`: Number of epochs with no improvement after which learning rate will be reduced.

   ```python
   rop_callback = ReduceLROnPlateau(monitor="val_loss", patience=3)
   ```

6. **Model Compilation**
   - **Purpose**: Compile the model with specified optimizer, loss function, and evaluation metrics.
   - **Parameters**:
     - `optimizer`: Optimization algorithm (RMSprop).
     - `loss`: Loss function (Categorical Crossentropy).
     - `metrics`: List of metrics to evaluate the model's performance.

   ```python
   METRICS = [tf.keras.metrics.CategoricalAccuracy(name='acc'),
              tf.keras.metrics.AUC(name='auc'),
              tfa.metrics.F1Score(num_classes=4)]
   CALLBACKS = [my_callback, rop_callback]
   
   custom_inception_model.compile(optimizer='rmsprop',
                                  loss=tf.losses.CategoricalCrossentropy(),
                                  metrics=METRICS)
   ```

7. **Model Summary**
   - **Purpose**: Display the architecture of the model, including the layers and their parameters.

   ```python
   custom_inception_model.summary()
   ```

![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/ef8b9a85-e9f3-4c82-bbe8-2cb90ccb7601)
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/a4b0beb5-cfd9-4b9f-91fb-0c8ef1be8cd4)
```python
fig, ax = plt.subplots(1, 3, figsize = (30, 5))
ax = ax.ravel()

for i, metric in enumerate(["acc", "auc", "loss"]):
    ax[i].plot(history.history[metric])
    ax[i].plot(history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/a6d1201f-add4-46b5-9e68-1ce56b50ec48)

### Results from Inception V3
```python
model2 = custom_inception_model
train_scores = model2.evaluate(train_data, train_labels)
val_scores = model2.evaluate(val_data, val_labels)
test_scores = model2.evaluate(test_data, test_labels)
print("Training Accuracy: %.2f%%"%(train_scores[1] * 100))
print("Validation Accuracy: %.2f%%"%(val_scores[1] * 100))
print("Testing Accuracy: %.2f%%"%(test_scores[1] * 100))
```
![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/2558ed8c-bc2f-480d-9262-57467e5bc051)

**Classification Report**
```python
CLASSES = [ 'NonDemented',
            'VeryMildDemented',
            'MildDemented',
            'ModerateDemented']

pred_labels = model2.predict(test_data)

def roundoff(arr):
    arr[np.argwhere(arr != arr.max())] = 0
    arr[np.argwhere(arr == arr.max())] = 1
    return arr

for labels in pred_labels:
    labels = roundoff(labels)

print(classification_report(test_labels, pred_labels, target_names=CLASSES))
```
![image](https://github.com/user-attachments/assets/0f23f25f-a3d5-4117-9b23-cf4c90246e82)

**Balanced Accuracy Score and Matthew's Correlation Coefficient**

**Balanced Accuracy** Score is a metric used to evaluate the performance of a classification model, especially in the presence of imbalanced datasets. It is defined as the average of sensitivity (true positive rate) and specificity (true negative rate). This metric is particularly useful when the classes are imbalanced, as it gives equal weight to both classes, ensuring that the performance on the minority class is not overshadowed by the majority class.

**Matthew’s Correlation Coefficient (MCC)** is another robust metric for evaluating the performance of binary classification models. It takes into account all four quadrants of the confusion matrix: true positives, true negatives, false positives, and false negatives. MCC is particularly valued for its ability to provide a balanced measure even when the classes are imbalanced. The coefficient ranges from -1 to +1, where +1 indicates perfect prediction, 0 indicates no better than random prediction, and -1 indicates total disagreement between prediction and observation. 

```python
print("Balanced Accuracy Score: {} %".format(round(BAS(test_ls, pred_ls) * 100, 2)))
print("Matthew's Correlation Coefficient: {} %".format(round(MCC(test_ls, pred_ls) * 100, 2)))
```
![image](https://github.com/user-attachments/assets/1dd60eac-8da9-4cb8-b9f4-3c013278a297)

![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/16a9294e-7535-49ff-b661-f8fa26eb2cbb)

<hr>



