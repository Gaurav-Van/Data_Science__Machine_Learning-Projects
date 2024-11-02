# Toxic Comment Classifier Using NLP and ML

Using Natural Language Processing, `nltk` and Machine Learning to classify comments into Six different Categories including their Neutral Cases. Using Streamlit and Langchain to create interactive web experience. 

- Toxic 
- Severe Toxic
- Threat 
- Obscene
- Insult
- Identity Hate
  
`Instead of Multiclass classification, Binary Classification of Each Category is performed`

[![Watch the video](https://img.youtube.com/vi/4SfdxNDU6DQ/hqdefault.jpg)](https://www.youtube.com/embed/4SfdxNDU6DQ)

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

### VECTORIZATION: Using TF-IDF and Unigram Approach

**Vectorization** is a key process in natural language processing (NLP) that converts text data into numerical vectors for machine learning models. The **TF-IDF (Term Frequency-Inverse Document Frequency)** approach, combined with the **unigram** model, is a popular method for this. TF-IDF evaluates the importance of words by considering their frequency in a document (TF) and their rarity across a corpus (IDF), thus highlighting significant terms while downplaying common ones. The unigram model treats each word as a single feature, capturing the frequency and importance of individual words without considering their order or context. This combination is particularly effective for tasks like text classification and sentiment analysis, where understanding the relevance of specific words can significantly enhance model performance. Would you like to explore more about other vectorization techniques or see some code examples?

**Function to perform Vectorization and model building**

Model Used For Each Category: KNN, Logistic Regression, SVM, CNB, BNB, DT and RF

```python
def vector_model(df, category, vectorizer, ngram):
    X = df['comment_text'].fillna(' ')
    Y = df[category]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    vector = vectorizer(ngram_range=(ngram), stop_words='english')

    X_train_scal = vector.fit_transform(X_train)
    X_test_scal = vector.transform(X_test)
    
    #KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scal, Y_train)
    Y_pred_knn = knn.predict(X_test_scal)
    print(f"Knn done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_knn)} ")
    print("\n----------------------------------------------------------------------")

    #logistic regression
    lr = LogisticRegression()
    lr.fit(X_train_scal, Y_train)
    Y_pred_lr = lr.predict(X_test_scal)
    print(f"\nLr done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_lr)} ")
    print("\n----------------------------------------------------------------------\n")

    #Support Vector Machine
    svm = SVC(kernel='rbf')
    svm.fit(X_train_scal, Y_train)
    Y_pred_svm = svm.predict(X_test_scal)
    print(f"\nsvm done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_svm)} ")
    print("\n----------------------------------------------------------------------\n")

    #Naive Bayes
    cnb = ComplementNB()
    cnb.fit(X_train_scal, Y_train)
    Y_pred_cnb = cnb.predict(X_test_scal)
    print(f"\ncnb done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_cnb)} ")
    print("\n----------------------------------------------------------------------\n")

    bnb = BernoulliNB()
    bnb.fit(X_train_scal, Y_train)
    Y_pred_bnb = bnb.predict(X_test_scal)
    print(f"\nbnb done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_bnb)} ")
    print("\n----------------------------------------------------------------------\n")

    #Decision Tree Classifier
    dt = DecisionTreeClassifier(criterion='entropy', min_samples_split=2, random_state=42)
    dt.fit(X_train_scal, Y_train)
    Y_pred_dt = dt.predict(X_test_scal)
    print(f"\nDT done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_dt)} ")
    print("\n----------------------------------------------------------------------\n")

    #Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=105, min_samples_split=2, random_state=42)
    rf.fit(X_train_scal, Y_train)
    Y_pred_rf = rf.predict(X_test_scal)
    print(f"\nRF done -> It's classification report for {category} category \n {classification_report(Y_test, Y_pred_rf)} ")
    print("\n----------------------------------------------------------------------\n")

    f1_scores = [round(f1_score(Y_pred_knn, Y_test), 2), round(f1_score(Y_pred_lr, Y_test), 2), round(f1_score(Y_pred_svm, Y_test), 2),
                 round(f1_score(Y_pred_cnb, Y_test), 2), round(f1_score(Y_pred_bnb, Y_test), 2), round(f1_score(Y_pred_dt, Y_test), 2),
                 round(f1_score(Y_pred_rf, Y_test), 2)]
    print(f"F1_scores for {category} category Are calculated")

    Scores = {f'F1_Score - {category}':f1_scores}
    Scores_df = pd.DataFrame(Scores, index=['KNN', 'Logistic Regression', 'SVM', 'Complement NB', 'Bernoulli NB', 'Decision Tree', 'Random Forest'])
    return Scores_df
```

#### F1 Scores of One of the Category: Toxic

![image](https://github.com/user-attachments/assets/519c180f-4b28-4c01-871a-bcd785488510)

#### Visualization of F1-Score of all Categories
```python
# Visualization of F1-Score of all categories
result = pd.concat([result_toxic, result_severe_toxic, result_threat, result_obscene, result_insult, result_identity_hate], axis=1)
result = result.transpose()
plt.figure(figsize=(15,15))
sns.lineplot(data=result, markers=True)
plt.legend(loc='best')
```
![image](https://github.com/user-attachments/assets/ac6d26bf-185e-4e28-b624-76815c11a3de)

Model Selected: Logistic Regression
- Exporting Trained ML Models as 6 pickle files [ one of each category ] 
- Exporting Trained Vectorized Models as 6 pickle files [ one for each category ]
  ```python
  def getfiles(df, label):
    x = df.comment_text.fillna(' ')
    y = df[label]
    
    tfv_f = TfidfVectorizer(ngram_range=(1,1), stop_words='english')
    X_vect = tfv_f.fit_transform(x)
    
    with open(f'{label + "_vect"}.pkl', 'wb') as f:
        pickle.dump(tfv_f, f)
    
    log = LogisticRegression()
    log.fit(X_vect, y)
    
    with open(f'{label + "_model"}.pkl', 'wb') as f:
        pickle.dump(log, f)
  list_c = ['toxic', 'severe_toxic', 'threat', 'obscene', 'insult', 'identity_hate']
  list_d = [df_toxic, df_severe_toxic, df_threat, df_obscene, df_insult, df_identity_hate]
  for i, j in zip(list_d, list_c):
      getfiles(i, j)
  ```
<hr>

## Building web app with the help of streamlit and langchain. Deploying it on Streamlit cloud
```python
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage
```

Streamlit is an open-source Python framework that simplifies the process of creating and sharing interactive web applications. Designed 
with data scientists and machine learning engineers in mind, Streamlit allows you to build data-driven apps with minimal code. Its intuitive 
API lets you add widgets, charts, and other interactive elements effortlessly, making it an excellent tool for quickly prototyping and deploying data apps.

LangChain is a powerful library for building applications that leverage large language models (LLMs). It provides tools to manage and orchestrate interactions
between users and AI models. Two key components of LangChain are the AIMessage and HumanMessage classes. HumanMessage represents messages from users, capturing
their inputs and queries. AIMessage, on the other hand, represents responses generated by the AI model.

**Langchain and Streamlit Introduction Configuration after importing models**
```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
         AIMessage(content="Hello! I serve as a Content Moderator. My role is to evaluate your comments for toxicity across six distinct categories. Rest assured, I am proficient in my job")
    ]

st.title("Toxic Comment Classifier Using NLP and ML")
image = Image.open('Image.jpg')
st.image(image, use_column_width=True)
```
**Function to Classify Comment into 6 Toxic Category**
```python
def toxic_classify(Input):

    classifications = []

    if Input is None or len(Input) == 0:
        return
    # toxic
    vect = toxic.transform([Input])
    zero = toxic_model.predict_proba(vect)[:, 0][0]
    one = toxic_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
        classifications.append('Neutral for Toxic Category')
    elif one > 0.58:
        classifications.append('Toxic')
    else:
       classifications.append('Non Toxic')

    # severe_toxic
    vect = severe_toxic.transform([Input])
    zero = severe_toxic_model.predict_proba(vect)[:, 0][0]
    one = severe_toxic_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
        classifications.append('Neutral for Severe Toxic Category')
    elif one > 0.58:
        classifications.append('Severe toxic')
    else:
       classifications.append('Non Severe toxic')

    # threat
    vect = threat.transform([Input])
    zero = threat_model.predict_proba(vect)[:, 0][0]
    one = threat_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
       classifications.append('Neutral for Threat Category')
    elif one > 0.58:
        classifications.append('Threat')
    else:
        classifications.append('Non Threat')

    # obscene
    vect = obscene.transform([Input])
    zero = obscene_model.predict_proba(vect)[:, 0][0]
    one = obscene_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
       classifications.append('Neutral for Obscene Category')
    elif one > 0.58:
        classifications.append('Obscene')
    else:
       classifications.append('Non Obscene')

    # insult
    vect = insult.transform([Input])
    zero = insult_model.predict_proba(vect)[:, 0][0]
    one = insult_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
        classifications.append('Neutral for Insult Category')
    elif one > 0.58:
        classifications.append('Insult')
    else:
         classifications.append('Non Insult')

    # identity_hate
    vect = identity_hate.transform([Input])
    zero = identity_hate_model.predict_proba(vect)[:, 0][0]
    one = identity_hate_model.predict_proba(vect)[:, 1][0]
    if (zero >= 0.42 and one <= 0.58) and (zero <= 0.58 and one >= 0.42):
         classifications.append('Neutral for Identity hate Category')
    elif one > 0.58:
         classifications.append('Identity hate')
    else:
         classifications.append('Non Identity hate')
    return classifications
```
**Langchain Sessions for AI and Human Messages**
```python
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content, unsafe_allow_html=True)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    result = toxic_classify(user_query)

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        try:
            response = f"Hmm.. According to my analysis, your comment <span style='color:yellow'><i>{user_query}</i></span> can be classified as: \n\n" + "\n".join(f"- {i}" for i in result)
            st.markdown(response, unsafe_allow_html=True)
            st.session_state.chat_history.append(AIMessage(content=response))
        except Exception as e:
            st.warning("I am Sorry. It looks like I made some mistake while trying to classify. Please Try Again. I will try my best")
            st.warning("Tip: Check for any mistake in the comment. Also Supported Language: English")
```

<hr>





