# Bangalore House Price Prediction
This Project predicts the price of a Real Estate property on the basis of Features like: `area_type`, `location`, `total_sqft`, `balcony`, `bathroom` and `BHK`

https://gaurav-van-house-price-predictor-streamlit-heroku-app-g56zmy.streamlitapp.com/

<hr>

## Libraries Required
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
%matplotlib inline 
#stored within the notebook
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20, 10)
import warnings
warnings.filterwarnings('ignore')
```
<hr>

## Data Collection 

Data From Kaggle: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data
```python
Data_frame = pd.read_csv("Bengaluru_House_Data.csv", na_values=[' ?'])
Data_frame.head(10)
```
![image](https://github.com/user-attachments/assets/9211a68c-061f-4f25-8f8f-8ed17b9e72e1)

<hr>

## Data Pre-Procesing

### Removing Not-so-important columns
In predicting Price of a property, 'availability' and 'society' are not THAT important factor in comparison to other features
```python
Data_frame_copy = Data_frame.copy()
Data_frame.drop(['availability', 'society'], axis=1, inplace=True)
```
### Checking, [removing or replacement] of null values and Data Formatting
```python
Data_frame.isnull().sum()
```
![image](https://github.com/user-attachments/assets/e93ef54d-2bbf-4cdd-8d44-87f1fccfc971)

```txt
Problem 1: 1, 16, 73 are negligible amount in front of 13320 enteries so we can drop,
in the case of balcony -> 609 we might wanna replace them with either mean or median depending on outliers
```
```python
# Droping Missing Values 
Data_frame.dropna(subset = ['location', 'size', 'bath'], inplace=True)
Data_frame.shape

#Replacing Missing Values
sns.boxplot(x='balcony', data=Data_frame)
plt.title("Balcony-BoxPlot")
plt.show()
#Replacing the missing data with mean 
Data_frame['balcony'].replace(np.nan, Data_frame['balcony'].mean(), inplace=True)
```

```txt
Problem 2: In size feature, some enteries are in the form of 2 BHK and some are in 2 bedrooms ,
Both are same so we can solve this by taking only the numeric value from the size feature
```
```python
Data_frame['BHK'] = Data_frame['size'].apply(lambda x: int(x.split(' ')[0]))
Data_frame.drop(['size'], axis=1, inplace=True)
Data_frame.head()
```

```txt
Problem 3: total_sqft should be float or numeric but here it is object [ because of dim1-dim2 input ] [and inputs like
sq Yards, Grounds, etc..] The data is Not Structured

Solution: so the best way would be to find an avg value of all dim1-dim2 input and replace it
and convert the different types of meaurement to sqft while doing so, all the values going in the
total_sqft feature will make the data type of that feature -> float
```
- **Creating Function to Convert ranges to a single standard value :** Sq.Meter, Perch, Sq.Yards, Acres, Cents, Guntha, Grounds. `1 Sq.Meter = 10.76 sqft`, `1 Perch = 272.25 sqft`, `1 Sq.Yard = 9 sqft`, `1 Acre = 43560.04 sqft`, `1 Cent = 435.56 sqft`, `1 Guntha = 1089 sqft` and `1 Ground = 2400.35 sqft`

- **Adding a Column / Feature which is important for future use [outlier treatment ] + gives a proper insight of a property**
  ```python
  Data_frame['price_per_sqft'] = Data_frame['price']*100000 / Data_frame['total_sqft']
  ```
### Outlier detection using Box plot, Outlier treatment using Flooring and Capping 



### Adding new data on the basis of Domain Knowledge

<hr>

<b>3. EDA -</b> Performing Data analysis on the basis of Domain Knowledge [ do check the jupyter file ] 
</br></br>
<b>4. Model Building</b><br><br>

* Encoding 
* As i am dealing with Regression problem, that too linear models so no need of Feature Scalling 
* Dividing the data by Train test split
* Testing Model's Score on divided data [ train_test_split and cross_val_score]
* <b>Model Used</b> - Linear Regression (Multiple Linear Regression)
</br></br>

<b>5. Deployment - </b> Building web app with the help of streamlit and deploying it on heroku cloud
<hr>
The Project / Web App is built in Python using the Following Libraries:
</br></br>

 * numpy
 * pandas
 * matplotlib
 * seaborn
 * sklearn
 * pickle
 * flask
 * streamlit
 * json
