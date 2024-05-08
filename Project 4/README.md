# Customized DL Model for Alzheimers Detection from MRI Images

**This Project Aims to Develop a Customized Deep Learning Model which Detects 4 Classes of Alzheimer (Non Demented,
Very Mild Demented, Mild Demented and Moderate Demented) from MRI Images. The goal is to improve early detection and
diagnosis of Alzheimer’s disease using non-invasive and accessible imaging technology**

<hr> 

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

## Data
Goal is to Train a Deep Learning Model on this type of Data. The trained model will be able to detect different classes of alzheimer from MRI Images. 

![image](https://github.com/Gaurav-Van/Data_Science__Machine_Learning-Projects/assets/50765800/7d1521eb-1394-4b03-87de-39a7c64c5cda)

<hr>

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




