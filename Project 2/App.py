import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import subprocess
import os
import base64
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes " \
                  "./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href


# Model building
def build_model(input_data):
    # Reads in saved regression model
    load_model = pickle.load(open('AChE_model_pubchem.pkl', 'rb'))
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)
    st.header('**Prediction Graph**')
    st.latex('pIC50 = -log (IC50)')
    fig = plt.figure()
    ax = sns.barplot(y=df['molecule_name'], x=df['pIC50'], errwidth=0)
    temp = int(max(df['pIC50'])) + 2
    plt.xticks(np.arange(0, temp, 0.5))
    plt.xlim([0, temp])
    for i in ax.containers:
        ax.bar_label(i, )
    plt.ylabel(" CHEMBL_ID ", size=12, fontstyle='normal', weight=600)
    plt.xlabel("pIC50 Values ", size=12, fontstyle='normal', weight=600)
    plt.title("pIC50 value of various Molecules", fontstyle='normal', weight=600)
    st.pyplot(fig)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Logo image
image = Image.open('logo.png')
st.image(image, use_column_width=True)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Page title
st.markdown ("""
# `Bioactivity Prediction Web App`
### *Acetylcholinesterase* 
This app allows you to predict the bioactivity towards inhibiting the Human `Acetylcholinesterase` enzyme. `Acetylcholinesterase` is a drug target for *Alzheimer's* disease.

### Some Important Information 

- ##### Fingerprints used: PubChem Fingerprints

- ###### Model Used: Random Forest Regression

- ###### R square value = 20-25 % :cry: :disappointed: But.. It tells us that even thought it, 
- Contains Significant features still it is  not able to explain variability in the target variable which further proves the alternative hypothesis of this Project. 

- This web app is just the demonstration of Prediction of Bioactivity `[ pIC50 = −log(IC50) ]` towards inhibiting Human AChE Enzyme Through Fingerprints, 
  Which is very poor Hence they are not the `Major Contributors` in this case.
 
- Descriptor calculated using [PaDEL-Descriptor](http://www.yapcwsoft.com/dd/padeldescriptor/) .
- [Read the Paper](https://doi.org/10.1002/jcc.21707).
 
- In short - Shit accuracy Model which gives us an very important Insight - *Low R square values are not always bad* :smile:, *in the field of research and analysis they can be useful
  to provide us with some insights* and, 
  it also predicts pIC50 value with less error and randomness of 75-80 percent.
  
---
""")

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](https://drive.google.com/file/d/1BkljjkCckFPn1mKLXIBsAmLk9d-ZBQ1R/view?usp=sharing)
""")

if st.sidebar.button('Predict'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)
    load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subset of descriptors from previously built model**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)
else:
    st.info('Upload input data in the sidebar to start!')
