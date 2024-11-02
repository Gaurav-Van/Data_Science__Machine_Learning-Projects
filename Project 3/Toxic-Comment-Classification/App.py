import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage

with open('toxic_vect.pkl', 'rb') as f:
    toxic = pickle.load(f)
with open('toxic_model.pkl', 'rb') as f:
    toxic_model = pickle.load(f)
with open('severe_toxic_vect.pkl', 'rb') as f:
    severe_toxic = pickle.load(f)
with open('severe_toxic_model.pkl', 'rb') as f:
    severe_toxic_model = pickle.load(f)
with open('threat_vect.pkl', 'rb') as f:
    threat = pickle.load(f)
with open('threat_model.pkl', 'rb') as f:
    threat_model = pickle.load(f)
with open('obscene_vect.pkl', 'rb') as f:
    obscene = pickle.load(f)
with open('obscene_model.pkl', 'rb') as f:
    obscene_model = pickle.load(f)
with open('insult_vect.pkl', 'rb') as f:
    insult = pickle.load(f)
with open('insult_model.pkl', 'rb') as f:
    insult_model = pickle.load(f)
with open('identity_hate_vect.pkl', 'rb') as f:
    identity_hate = pickle.load(f)
with open('identity_hate_model.pkl', 'rb') as f:
    identity_hate_model = pickle.load(f)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
         AIMessage(content="Hello! I serve as a Content Moderator. My role is to evaluate your comments for toxicity across six distinct categories. Rest assured, I am proficient in my job")
    ]

st.title("Toxic Comment Classifier Using NLP and ML")
image = Image.open('Image.jpg')
st.image(image, use_column_width=True)

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

user_query = st.chat_input("Type your comment...")

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
