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

Outliers are those data points those are way off from our main data set [ abnormal data points ] Now they can be of `Type1-> Data points / numerical` and 
`Type2 -> Domain error [ abnormality in the Domain Knowledge ]`. Type1 and Type2 are similar, Not that Different

- **Fixing Type 1 Outliers**: Problem in total_sqft, bathroom, price, BHK and price_per_sqft
  
  ```python
  # Applying Quantile Based Flooring and capping
  lower_bound = Data_frame['total_sqft'].quantile(0.10)
  upper_bound = Data_frame['total_sqft'].quantile(0.90)
  Data_frame['total_sqft'] = np.where(Data_frame['total_sqft'] < lower_bound, lower_bound, Data_frame['total_sqft'])
  Data_frame['total_sqft'] = np.where(Data_frame['total_sqft'] > upper_bound, upper_bound, Data_frame['total_sqft'])

  # Bathroom - small quanitites of Outliers so Replace them with median 
  median = Data_frame['bathroom'].quantile(0.50)
  upper_out = Data_frame['bathroom'].quantile(0.95)
  Data_frame['bathroom'] = np.where(Data_frame['bathroom'] > upper_out, median, Data_frame['bathroom'])

  # Applying Quantile Based Flooring and capping
  lower_bound = Data_frame['price'].quantile(0.10)
  upper_bound = Data_frame['price'].quantile(0.90)
  Data_frame['price'] = np.where(Data_frame['price'] < lower_bound, lower_bound, Data_frame['price'])
  Data_frame['price'] = np.where(Data_frame['price'] > upper_bound, upper_bound, Data_frame['price'])
  ```
- **Fixing Type 2 Outliers**

  In a General Real Esate Property / House, the number of Bathrooms depends on number of Bedrooms [BHK]. The equations in general is
  total Bathroom <= BHK + 1 [1 - extra for Guest] It is unusual to have 2 more bathrooms than number of bedrooms in a home

  ```python
  Data_frame = Data_frame[(Data_frame['bathroom'] < (Data_frame['BHK'] + 2))]
  Data_frame.shape
  Data_frame['balcony'] = Data_frame['balcony'].astype('int')
  ```
  Problem here is that in some cases - price of 2bhk is more than price of 3bhk for similar sqft area and same location and hence is a
  unexpected error or outlier. Here Prices of 2bhk are more than 3bhk for similar or same total_sqft area, it means price_per_sqft of 2bhk
  should be more than 3bhk for the same location and similar total_sqft area So now we can remove those 3BHK whose price_per_sqft is less than
  mean price_per_sqft of 2 BHK. We here are dealing with same location [ because Different Location will affect the price ]

  ```python
  def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis=0)
  Data_frame = remove_bhk_outliers(Data_frame)
  Data_frame.shape
  ```
  In general Square ft per Bedroom is 300 anything less than that is suspicious and can be declared as outlier

  ```python
  Data_frame=Data_frame[~(Data_frame.total_sqft/Data_frame.BHK<300)]
  Data_frame.shape
  ```

<hr>

## EDA: Performing Data analysis on the basis of Domain Knowledge

### Finding Correlation between Independent and dependent Variables / features

```python
num_corr = Data_frame[['total_sqft', 'bathroom', 'balcony', 'BHK']].corr()
sns.heatmap(num_corr, cmap='RdBu')
plt.title("Heatmap of Correlation between numeric Independent Features")
plt.show()
```
![image](https://github.com/user-attachments/assets/2923f6b8-f957-48f6-b2c8-7ceb27f5e43f)

The P value from pearson correlation is 0 ( < 0.05 ) so our Correlation coef are statically significant balcony the weakest while total_sqft the strongest
Correlation is not equal to causation -> Balcony might be weakest but is an important feature in deciding the price of a Property

### Let's Understand Categorical Features 

**Average price of category from categorical features**

```python
df3 = Data_frame[['area_type','location', 'price']]
df_p = df3.groupby(['area_type','location'], as_index=False).mean().sort_values(ascending=False, by='price')
df_p.set_index('area_type', inplace=True)
df_p.rename(columns={'price':'avg-price'}, inplace=True)
df_pivot = pd.pivot_table(df_p, index='area_type', columns='location', values='avg-price')
df_pivot.fillna(0, inplace=True)
df_pivot 
```
![image](https://github.com/user-attachments/assets/1cda96af-6bf3-4a9f-b4dd-b4b8bde7c42b)

<hr>

## Model Building

### Encoding 
Encoding the Categoricals features - area_type and location
```python
one_dum = pd.get_dummies(Data_frame['area_type'])
Data_frame = pd.concat([Data_frame, one_dum], axis=1)
Data_frame.drop(['area_type'], axis=1, inplace=True)
ne_dum = pd.get_dummies(Data_frame['location'])
Data_frame = pd.concat([Data_frame, ne_dum], axis=1)
Data_frame.drop(['location'], axis=1, inplace=True)
```
**As I am dealing with Regression problem, that too linear models so feature scalling is ignored here**

### Dividing the data by Train test split
![image](https://github.com/user-attachments/assets/fe79989a-a77e-4354-a7ed-c1b78b5e76b1)

### Multiple Linear Regression Model
```python
LinearModel = LinearRegression()
LinearModel.fit(X_train, Y_train)
Y_pred = LinearModel.predict(X_test)
Y_pred
```
- R^2 Value: 0.824561107732018
- MSE Value: 669.3971397935138
- RMSE Value: 25.87271032

![image](https://github.com/user-attachments/assets/3fa926e0-e281-4e49-a62c-f031c691611c)

### Decision tree for Regression
```python
Tree_model = DecisionTreeRegressor(max_depth=5)
Tree_model.fit(X_train, Y_train)
Y_hat = Tree_model.predict(X_test)
```
- R^2 Value: 0.8004133365466866

### Random Forest 
```python
rf = RandomForestRegressor(n_estimators=105)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
```
- R^2 Value: 0.8414810182841019

### Lazy Predict 
```python
import lazypredict
from lazypredict.Supervised import LazyRegressor
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)
```
### Exporting Model and Columns
```python
import pickle
with open('bangalore_home_prices_model.pickle', 'wb') as obj:
    pickle.dump(LinearModel, obj)

import json
columns = {
    'Columns': [col.lower() for col in X.columns]
}
with open("Columns.json", 'w') as f:
    f.write(json.dumps(columns))
```

<hr>

## Deployment Building web app with the help of streamlit and deploying it on streamlit cloud

### Importing Model and Columns
```python
with open(
        r"C:\Users\Asus\PycharmProjects\Real_Estate_price_prediction\Model\bangalore_home_prices_model.pickle", 
        'rb') as f:
    __model = pickle.load(f)

with open(r"C:\Users\Asus\PycharmProjects\Real_Estate_price_prediction\Model\Columns.json", 'r') as obj:
    __data_columns = json.load(obj)["Columns"]
    __area_types = __data_columns[4:8]
    __locations = __data_columns[8:]
```
### Prediction Function
```python
def get_predicted_price(area_type, location, sqft, balcony, bathroom, BHK):
    try:
        area_index = __data_columns.index(area_type.lower())
        loc_index = __data_columns.index(location.lower())
    except ValueError as e:
        area_index = -1
        loc_index = -1

    lis = np.zeros(len(__data_columns))
    lis[0] = sqft
    lis[1] = bathroom
    lis[2] = balcony
    lis[3] = BHK

    if loc_index >= 0 and area_index >= 0:
        lis[area_index] = 1
        lis[loc_index] = 1

    price = round(__model.predict([lis])[0], 2)
    strp = ' lakhs'

    if math.log10(price) >= 2:
        price = price / 100
        price = round(price, 2)
        strp = " crores"

    return str(price) + strp
```
### Streamlit Web App
```python
def main():
    global result
    st.title("Bangalore House Price Predictor")
    html_temp = """
           <div>
           <h2>House Price Prediction ML app</h2>
           </div>
           """
    st.markdown(html_temp, unsafe_allow_html=True)
    total_sqft = st.text_input("Total_sqft")
    balcony = st.text_input("Number of Balconies")
    bathroom = st.text_input("Number of Bathrooms")
    BHK = st.text_input("BHK")
    area_type = st.selectbox("Area Type", __area_types)
    location = st.selectbox("Location", __locations)

    if st.button("Predict"):
        result = get_predicted_price(area_type, location, total_sqft, balcony, bathroom, BHK)

    st.success(f"Price = {result}")
```
<hr>
