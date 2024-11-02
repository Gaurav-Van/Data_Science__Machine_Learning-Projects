# Toxic Comment Classifier Using NLP and ML

Using Natural Language Processing, `nltk` and Machine Learning to classify comments into Six different Categories including their Neutral Cases. Using Streamlit and Langchain to create interactive web experience. 

- Toxic 
- Severe Toxic
- Threat 
- Obscene
- Insult
- Identity Hate
  
`Instead of Multiclass classification, Binary Classification of Each Category is performed`

Web App Link - https://gaurav-van-toxic-comment-web-app-app-24y37c.streamlitapp.com/
<hr>

## Libraries and Data Collection 

**From Kaggle:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
```
```python
df_train = pd.read_csv('train.csv', na_values=[' ?'])
df_test = pd.read_csv('test.csv')
df_train['comment_text'].fillna(' ')
df_test['comment_text'].fillna(' ')
```
![image](https://github.com/user-attachments/assets/c4e85c56-7f33-4318-b9e4-0b3455d395e4)

<hr>

## Data Pre-Procesing - Text Pre-Processing Using Regular Expressions

- Removing \n characters 
- Removing Aplha-Numeric Characters
- Removing Punctuations
- Removing Non Ascii Characters

```python
import re
import string

remove_n = lambda x: re.sub("\n", "", x)

remove_alpha_num = lambda x: re.sub("\w*\d\w*", '', x)

remove_pun = lambda x: re.sub(r"([^\w\s]|_)", '', x.lower())

remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)

df_train['comment_text'] = df_train['comment_text'].map(remove_n).map(remove_alpha_num).map(remove_pun).map(remove_non_ascii)
df_test['comment_text'] = df_test['comment_text'].map(remove_n).map(remove_alpha_num).map(remove_pun).map(remove_non_ascii)
```
<hr>

## EDA - Performaing Data analysis to Discover Potential Issues and trend of the Data

![image](https://github.com/user-attachments/assets/9a2f11e3-ee8f-4c44-a376-a3554da579ee)

Through Bar charts of Each Category: <b>Prob</b> = Class Imbalance. <b>Solution</b> = Making Frequency of 0s equal to Frequency of 1s by Making Different Dataset of each Category [ id, comment_text, category]. Helps to solve the Issue of Class Imbalance and Helps in Binary Classification of Each Category

- id, comment_text, toxic
- id, comment_text, severe_toxic
- id, comment_text, obscene
- id, comment_text, threat
- id, comment_text, insult
- id, comment_text, identity_hate

```python
df_toxic = df_train[['id', 'comment_text', 'toxic']]
df_severe_toxic = df_train[['id', 'comment_text', 'severe_toxic']]
df_obscene = df_train[['id', 'comment_text', 'obscene']]
df_threat = df_train[['id', 'comment_text', 'threat']]
df_insult = df_train[['id', 'comment_text', 'insult']]
df_identity_hate = df_train[['id', 'comment_text', 'identity_hate']]
```

**Making toxic category balance**

15294 = toxic [ from graph ] 15294 = non-toxic [ to balance out ]
```python
df_toxic_1 = df_toxic[df_toxic['toxic'] == 1]
df_toxic_0 = df_toxic[df_toxic['toxic'] == 0].iloc[:15294]
df_toxic_bal = pd.concat([df_toxic_1, df_toxic_0], axis=0)
```

**Making severe_toxic category balance**

1595 = severe toxic [ from graph ] 1595 = non severe toxic [ to balance out ]
```python
df_severe_toxic_1 = df_severe_toxic[df_severe_toxic['severe_toxic'] == 1]
df_severe_toxic_0 = df_severe_toxic[df_severe_toxic['severe_toxic'] == 0].iloc[:1595]
df_severe_toxic_bal = pd.concat([df_severe_toxic_1, df_severe_toxic_0], axis=0)
```
**Making obscene category balance**

8449 = obscene [ from graph ] 8449 = non obscene [ to balance out ]
```python
df_obscene_1 = df_obscene[df_obscene['obscene'] == 1]
df_obscene_0 = df_obscene[df_obscene['obscene'] == 0].iloc[:8449]
df_obscene_bal = pd.concat([df_obscene_1, df_obscene_0], axis=0)
```
**Making threat category balance**

478 = threat [ from graph ] 478 = non threat [ to balance out ]|
```python
df_threat_1 = df_threat[df_threat['threat'] == 1]
df_threat_0 = df_threat[df_threat['threat'] == 0].iloc[:700]
df_threat_bal = pd.concat([df_threat_1, df_threat_0], axis=0)
```
**Making insult category balance**

7877 = insult [ from graph ] 7877 = non insult [ to balance out ]
```python
df_insult_1 = df_insult[df_insult['insult'] == 1]
df_insult_0 = df_insult[df_insult['insult'] == 0].iloc[:7877]
df_insult_bal = pd.concat([df_insult_1, df_insult_0], axis=0)
```
**Making identity_hate category balance**

1405 = identity hate [ from graph ] 1405 = non identity hate [ to balance out ]
```python
df_identity_hate_1 = df_identity_hate[df_identity_hate['identity_hate'] == 1]
df_identity_hate_0 = df_identity_hate[df_identity_hate['identity_hate'] == 0].iloc[:1405]
df_identity_hate_bal = pd.concat([df_identity_hate_1, df_identity_hate_0], axis=0)
```
### Analysing most frequent words using wordcharts
```python
def frequent_words(dataset, category):
    stopwords = STOPWORDS
    wc = WordCloud(width = 600, height = 600, random_state=42, background_color='black', colormap='rainbow', collocations=False, stopwords = stopwords)
    filter = dataset[dataset[category] == 1]
    text = filter.comment_text.values
    wc.generate(' '.join(text))
    wc.to_file(f"Frequent words in balanced classes/Frequent words in {category} category.png")
```
**Threat category**

![image](https://github.com/user-attachments/assets/e8dbbef3-1893-4bc8-9cfe-ad4ab155ea76)

<hr>

## Model Building

* <b>VECTORIZATION :-</b> Using TF-IDF and Unigram Approach
* <b>Model Used For Each Category :-</b> KNN, Logistic Regression, SVM, CNB, BNB, DT and RF
* <b>Model Selected/b> - Logistic Regression
* Exporting Trained ML Models as 6 pickle files [ one of each category ] 
* Exporting Trained Vectorized Models as 6 pickle files [ one for each category ] 
</br></br>

<hr>

## Building web app with the help of streamlit and langchain. Deploying it on Streamlit cloud

<hr>





