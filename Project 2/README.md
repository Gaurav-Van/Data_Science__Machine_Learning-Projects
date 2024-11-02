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
- Are they significant in explaining bioactivity of Compounds inhibiting Human AChE enzyme ?
- <i>They Do contribute but the curosity is, are they the Major Contributors or we should look for another ones when it comes to inhibiting Human AChE Enzyme</i> 
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
- Exploratory Data Analysis 
- Regression Concept of Machine Learning [<b> Regression Analysis </b> ]
- Streamlit for building the Web App
