Web App Link :- https://gaurav-van-toxic-comment-web-app-app-24y37c.streamlitapp.com/

# Toxic Comment Classifier Using NLP and ML

Classifying Comments in Six different Categories including their Neutral Cases Using Concepts of NLP and ML
- Toxic 
- Severe Toxic
- Threat 
- Obscene
- Insult
- Identity Hate
<hr>

## Concept Used</br></br>
Instead of Multiclass classification, Binary Classification of Each Category is performed</br></br>
<b>1. Data Collection -</b> From Kaggle: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge<br><br>
<b>2. Data Pre-Procesing - Text Pre-Processing Using Regular Expressions</b><br>
* Removing \n characters 
* Removing Aplha-Numeric Characters
* Removing Punctuations  
* Removing Non Ascii Characters

<b>3. EDA - Performaing Data analysis to Discover some Issues and trend of the Data</b><br>
 - Through Bar charts of Each Category :- <b>Prob</b> = Class Imbalance -> <b>Solution</b> = Making Frequency of 0s equal to Frequency of 1s by Making Different Dataset of each Category [ id, comment_text, category]. 
 - Helps to solve the Issue of Class Imbalance and Helps in Binary Classification of Each Category

<b>4. Model Building</b><br>

* <b>VECTORIZATION :-</b> Using TF-IDF and Unigram Approach
* <b>Model Used For Each Category :-</b> KNN, Logistic Regression, SVM, CNB, BNB, DT and RF
* <b>Model Selected/b> - Logistic Regression
* Exporting Trained ML Models as 6 pickle files [ one of each category ] 
* Exporting Trained Vectorized Models as 6 pickle files [ one for each category ] 
</br></br>

<b>5. Deployment - </b> Building web app with the help of streamlit and deploying it on Streamlit cloud
<hr>




