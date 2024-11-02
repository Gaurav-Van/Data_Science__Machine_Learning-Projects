#  Drug Discovery Using ML and EDA 

#### <b> Predicting Bioactivity towards inhibiting the Acetylcholinesterase enzyme - A drug target for Alzheimer's Disease </b> 

- ##### <b>One Line Summary of Project</b> - To Study Drug-Protein Interaction Based on their Fingerprints

- When the brain's structures begin to shrink, it can no longer efficiently produce the ACh chemical needed for intra-neuronal messaging and the receptors responsible for catching these chemicals to process the signal are also less plentiful. Although both of these components begin to fail simultaneously, the decline in the number of receptors to catch these ACh molecules is much more significant than the reduced chemical production.

- At this stage, although there no treatments can reverse the process of destruction, there are options to promote a strong signaling pathway for as long as possible. This is the goal of an <b>Acetylcholinesterase inhibitor</b>. It works by blocking an enzyme, acetylcholinesterase, from attaching to the ACh chemical and digesting it. By allowing the chemical to have more time to bind to a receptor, there is more of a chance for a message to be sent from one neuron to another and allow the body to have coordinated movements and functioning.

**Recent Discoveries in Drug Discovery of Alzheimer's:** https://www.bbc.com/news/health-63749586

<hr>

## The Ultimate Aim that I want to achieve

- To study the Chemical Structure [ here - canonical_smiles ] of Compounds inhibiting Human AChE enzyme with the help of Lipinski Descriptors

- To Study the Relantionship between  Molecular Fingerprints Descriptor andMolar Concentration [IC50] value of the Compound, inhibiting the Human AChE enzyme

- On the basis of Insights from the relationship [ be it weak or Strong, in both cases it will provide us with some insights ], Building a Model and a Web App which will predict the Bioactivity [ IC50 ] of the Compound
 
 ### The Problem Statement
```
Fingerprints are arrnagement of bits on the basis of Division of Chemical Structure of Compounds so Logically
they should not be Major Contributors in explaining the Variations in IC50 values [ Molar Concentration ] of
inhibiting Human AChE Enzyme
```
```
Fingerprints Holds an Important Place in the field of Bioinformatics and Genomics. They are Quantative Arrangement
of Structures on the basis of some rules and Algorithms, Explaining certain properties in that manner . They Holds an
important place in drug biology and are important in explaining bioacitivity of many different types of compounds but
now a question arises within me which is,
```
- Are they significant in explaining bioactivity of Compounds inhibiting Human AChE enzyme ? They Do contribute but the curosity is,
  are they the Major Contributors or we should look for another ones when it comes to inhibiting Human AChE Enzyme
  
- So this is what we are Trying to Study, the relationship between Fingerprints and Bioactivity [IC50] of Compounds, inhibiting the Human AChE enzyme 

**I wish to achieve this Relationship with the Help of the concept of Hypothesis Testing , here**

- <b>Alternative Hypothesis [Ha]</b> - Fingerprints are not Major Contributors in explaining the Variations in IC50 values inhibiting the Human AChE Enzyme
- <b>Null Hypothesis [H0]</b> - Fingerprints are Major Contributors in explaining the Variations in IC50 values inhibiting the Human AChE Enzyme [this is what I wish to disprove]

#### What I Wish to Achieve ?
##### A low R square value .. But Why ?
R square value in regression analysis tells us about how much variation in dependent / Target variable can be explained by Independent variables. A Relatively low value would suggest that even though the features are significant they are not able to explain the variation, which will support my cause and will result into rejecting null hypothesis
Will i get that Value? Let's Find Out 

<hr>

## Concept Used

### ChEMBL Databse
The ChEMBL Database is a database that contains curated bioactivity data of more than 2 million compounds. It is compiled from more than 76,000 documents, 1.2 million assays and the data spans 13,000 targets and 1,800 cells and 33,000 indications

### Libraries and Search for Target Protein
Installing ChEMBL web service Package to Extract bioactivity data from the ChEMBL Database
```python
! pip install chembl_webresource_client

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
import math
from numpy.random import seed
from numpy.random import randn
from scipy.stats import mannwhitneyu

target = new_client.target
target_query = target.search('Acetylcholinesterase')
target_query # Dictionary
targets = pd.DataFrame(target_query)
targets.columns
```
![image](https://github.com/user-attachments/assets/56296666-c089-4c6a-b9f2-8ace0e8d39fe)

**Out target is to Predict for Human Acetylcholinesterase:** Retrieve chembl id for Human Acetylcholinesterase ['CHEMBL220']
```python
filt = (targets.organism == 'Homo sapiens') & (targets.pref_name == 'Acetylcholinesterase')
targets[filt]
selected_target = targets[filt]['target_chembl_id'][0]
selected_target
```
![image](https://github.com/user-attachments/assets/4d972f67-8110-482a-94db-f573e5a9a6c2)

#### Potency
In the field of pharmacology, potency is a measure of drug activity expressed in terms of the amount required to produce an effect of given intensity
A highly potent drug (e.g., fentanyl, alprazolam, risperidone, bumetanide, bisoprolol) evokes a given response at low concentrations, while a drug of 
lower potency (meperidine, diazepam, ziprasidone, furosemide, metoprolol) evokes the same response only at higher concentrations

#### IC50
The half maximal inhibitory concentration (IC50) is a measure of the potency of a substance in inhibiting a specific biological or biochemical function.
IC50 is a quantitative measure that indicates how much of a particular inhibitory substance (e.g. drug) is needed to inhibit, in vitro, a given biological 
process or biological component by 50%. The biological component could be an enzyme, cell, cell receptor or microorganism. IC50 values are typically expressed as molar concentration.
```python
activity = new_client.activity
res = activity.filter(target_chembl_id = selected_target).filter(standard_types="IC50")
df = pd.DataFrame(res)
test = df[['standard_value', 'standard_type', 'standard_units', 'type', 'units']]
filt = df['standard_type'] == 'IC50' 
df = df[filt]
df.to_csv("AChE_Bioactivity_data_1.csv", index=False)
```
<hr>

### Data Pre-Processing

**Required Columns are :** `molecule_chembl_id` , `canonical_smiles` and `standard_value`
- molecule_chembl_id has no missing values - so we are good to go
- canonical_smiles contains 0.43% of missing values which we can drop easily as replacing values brings variability / variations so droping them is the best option
- standard_value contains 21.74% of missing values which we can drop easily as replacing values brings variability / variations so droping them is the best option

```python
df1.dropna(subset=['canonical_smiles', 'standard_value'], axis=0, inplace=True)
df2 = df1[['molecule_chembl_id','canonical_smiles', 'standard_value']]
df2.to_csv('AChE_Bioactivity_data_2.csv', index=False)
```

**Labeling Compounds as either being active, inactive or intermediate:** The Bioactivity data is in the IC50 unit. Compounds Having values [Potency] of
- Less than 1000nM will be considered as active
- Greater than 10000nM will be considered as inactive
- Between 1000nM and 10000nM will be considered as intermediate

```python
df3 = pd.read_csv('AChE_Bioactivity_data_2.csv')
bioactivity_threshold = []
for i in df3['standard_value']:
    if i >= 10000:
        bioactivity_threshold.append('inactive')
    elif i <= 1000:
        bioactivity_threshold.append('active')
    else:
        bioactivity_threshold.append('intermediate')
bioactivity_class = pd.Series(bioactivity_threshold, name='class')
df4 = pd.concat([df3, bioactivity_class], axis=1)
df4.to_csv('AChE_Bioactivity_data_3.csv', index=False)
df4['class'].value_counts()
```
![image](https://github.com/user-attachments/assets/d7de6c63-deb7-4e61-a249-2976bfaacdce)

#### Calculate Lipinski Descriptors

Christopher Lipinski, a scientist at Pfizer, came up with a set of rule-of-thumb for evaluating the druglikeness of compounds. Such druglikeness
is based on the Absorption, Distribution, Metabolism and Excretion (ADME) that is also known as the pharmacokinetic profile. Lipinski analyzed 
all orally active FDA-approved drugs in the formulation of what is to be known as the Rule-of-Five or Lipinski's Rule.

The Lipinski's Rule stated the following:

- Molecular weight < 500 Dalton
- Octanol-water partition coefficient (LogP) < 5
- Hydrogen bond donors < 5
- Hydrogen bond acceptors < 10

```python
def lipinski(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
       
        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)
           
        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MW","LogP","NumHDonors","NumHAcceptors"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors
df_lipinski = lipinski(df5['canonical_smiles'])
df6 = pd.concat([df5, df_lipinski], axis=1)
```
#### Converting IC50 to pIC50
To Reduce Skewness and Variablity in the Data and allowing Data to be uniformly Distributed, I will convert IC50 to pIC50 
Value which is basically a negative log -> -log10(IC50)

- Take the IC50 values from the standard_value column and converts it from nM to M by multiplying the value by 10^-9
- Take the molar value and apply -log10
- Delete the standard_value column and create a new pIC50 column

```python
norm =[]
for i in df7.standard_value:
    if i > math.pow(10, 8):
        i = math.pow(10, 8)
    norm.append(i)
df7['standard_norm_value'] = norm
df7.drop('standard_value', axis=1, inplace=True)
pIC50=[]
for i in df7.standard_norm_value:
    molar = i*math.pow(10, -9)
    pIC50.append(-np.log10(molar))
df7['pIC50'] = pIC50
df7.drop('standard_norm_value', axis=1, inplace=True)
```
**Removing Intermediate Bioactivity Class:** To perform Mann whitney U Test

<hr>

### Exploratory Data Analysis using Lipinski Descriptors

**Scatter plot of MW vs LogP**

![image](https://github.com/user-attachments/assets/7cf72242-4af1-4a21-94c1-dace580f5e6d)

#### Statistical analysis - Mann-Whitney U Test
The Mann-Whitney U Test is a statistical test used to determine if 2 groups are significantly different from each other 
on the basis of variable of interest. Variable of interest should be continuous and 2 groups should have similar values
on variable of interest.
```python
def mannwhitney(descriptor, verbose=False):
    
    selection = [descriptor, 'class']
    df_test = df8[selection]
    active = df_test[df_test['class'] == 'active']
    active = active[descriptor]

    selection = [descriptor, 'class']
    df_test = df8[selection]
    inactive = df_test[df_test['class'] == 'inactive']
    inactive = inactive[descriptor]
    
    stat, p = mannwhitneyu(active, inactive)

    alpha = 0.05
    if p > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distribution (reject H0)'
  
    results = pd.DataFrame({'Descriptor':descriptor,
                          'Statistics':stat,
                          'p':p,
                          'alpha':alpha,
                          'Interpretation':interpretation}, index=[0])
    filename = 'mannwhitneyu_' + descriptor + '.csv'
    results.to_csv(filename)

    return results
```

#### Interpretation of Statistical Results

`pIC50 values`

Taking a look at pIC50 values, the actives and inactives displayed statistically significant difference, which is to be expected since threshold values

- IC50 < 1,000 nM = Actives while IC50 > 10,000 nM = Inactives, corresponding to
- pIC50 > 6 = Actives and pIC50 < 5 = Inactives were used to define actives and inactives.
  
`Lipinski's descriptors`

3 of the 4 Lipinski's descriptors exhibited statistically significant difference between the actives and inactives.

#### Computing Molecular Descriptors

Molecular Descriptors are Quantitive Description of the Compounds, can be defined as mathematical representations of molecules 
properties that are generated by algorithms. The numerical values of molecular descriptors are used to quantitatively describe 
the physical and chemical information of the molecule Fingerprint is the Molecular Descriptor that is used here, Specifically 
pubchem Fingerprints. Molecular Fingerprint are useful in predicting drug Binding Affinities or Bioactivity

**Bit allocation in Pubchem Fingerprints**

![image](https://github.com/user-attachments/assets/129bdcf5-e9ea-43fe-bbde-41cfa24eeddf)

**Software used : PaDEL-Descriptor** `! wget https://github.com/dataprofessor/bioinformatics/raw/master/padel.zip`
```python
df_final = pd.read_csv('AChE_Bioactivity_data_4.csv')
df_final_selected = df_final[['canonical_smiles', 'molecule_chembl_id']]
df_final_selected.to_csv('molecule.smi', sep='\t', index=False, header=False)
```
**PaDEL-Descriptor** 

![image](https://github.com/user-attachments/assets/890d80d2-fe42-4e4a-a7d7-260255309ef2)
![image](https://github.com/user-attachments/assets/a725ef57-c450-440a-97ab-aec20a4905b5)

`Result obtained from the Software is Stored in Fingerprints.csv`

<hr>

### Regression Analysis of PubChem Fingerprints
_similar cases for other fingerprints_

#### Libraries Required
```python
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
import lazypredict
from lazypredict.Supervised import LazyRegressor
import lightgbm as ltb
import math
from scipy import stats
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
```
**Variance Threshold**

The Variance Threshold is a feature selection technique used in data analysis and machine learning to remove features with 
low variance. This method is particularly useful in simplifying models, reducing overfitting, and improving computational efficiency.

```python
def remove_low_variance(input_data, threshold):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = remove_low_variance(X, threshold=0.1)

p_value = []
col_name = []
names = []
for x in X.columns:
    w = stats.pearsonr(DataSet[x], Y)
    p_value.append(w[1])
    col_name.append(x)
Imp = pd.DataFrame({'Column_name':col_name,
                    'p-value':p_value})
filt = Imp['p-value'] > 0.05
df = Imp[filt]
for i in df['Column_name']:
    names.append(i)

X.drop(names, axis=1, inplace=True)
X.to_csv('descriptor_list.csv', index=False)
```
**Data Splitting**

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
print(f"{X_train.shape} \t {Y_train.shape} \n {X_test.shape} \t {Y_test.shape}")
```

**Lazy Regressor for Regression Analysis**

```python
clf = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
models_test,predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)
plt.figure(figsize=(5, 10))
sns.set_theme(style="whitegrid")
ax = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
ax.set(xlim=(0, 10))
```
![image](https://github.com/user-attachments/assets/00c801a8-da0c-4b75-9586-538a52863c0f)

```python
import pickle
pickle.dump(model, open('AChE_model_pubchem.pkl', 'wb'))
```
**Key Takeaways**

After Observing the Results from all 5 fingerprints
- Data is in form of 0s and 1s, so no place for encoding and Feature Scalling
- From the list from Lazy Predict and our random forest model with hyperparameter tuning, we can tell that,
- The range of R square values ranges from 20-28 percent, which is quite low
- The error / loss / MSE value ranges from 1-2, which is optimal

<hr>

### Streamlit for building the Web App

<hr>

