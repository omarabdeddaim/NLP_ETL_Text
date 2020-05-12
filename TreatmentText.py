# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 00:53:02 2020

@author: Omar Abdeddaim
"""

import pandas as pd

df = pd.read_csv("DataTotal1.csv")

R = df["remarque"]

def lev_dist(data1, data2):
    if data1 == data2:
        return 0
    # Prepare a matrix
    slen, tlen = len(data1), len(data2)
    dist = [[0 for i in range(tlen+1)] for x in range(slen+1)]
    for i in range(slen+1):
        dist[i][0] = i
    for j in range(tlen+1):
        dist[0][j] = j

    # Counting distance, here is my function
    for i in range(slen):
        for j in range(tlen):
            cout = 0 if data1[i] == data2[j] else 1
            dist[i+1][j+1] = min(
                            dist[i][j+1] + 1,   # deletion
                            dist[i+1][j] + 1,   # insertion
                            dist[i][j] + cout   # substitution
                        )
    return dist[-1][-1]
# Delete All Sans Objection & nan
Liste = []
for i in range(len(R)):
    print(i)
    Liste.append(lev_dist(str(L[i]),'techniquement sans objection'))
    
    
for i in range(len(R)):
    if lev_dist(str(L[i]),'techniquement sans objection') <=25:
        L[i] = 'Veuillez faire attention au côté administratif du projet et la validation des notes des articles concernés.'
    if (L[i]=='S.O' or L[i]=='S.O.' or L[i]=='S;O'):
        L[i] = 'Veuillez juste revoir vos étape et voir leurs conforme avec les norms du projet.'
    if (L[i] =='CGP' or L[i]=='.....' or L[i]=='????' or L[i]=='l' or L[i]=='jg' or L[i]=="???????????" or L[i]=='......' or L[i]=='A.M' or L[i]=='' or L[i]==' ' or L[i]=='....' or L[i]=='..'):                 
        L[i]="Veuillez vous revoir la clarté sur le cahier de charge"


df = df.drop(columns=["remarque_Code"])
Titre = [t for t in df]


#Clustering des Remarques
import numpy as np
df['remarque_Code'] = np.zeros(52135)

for i in range(len(df)):
    if df.iloc[i,27] == 'Veuillez faire attention au côté administratif du projet et la validation des notes des articles concernés.':
        df.iloc[i,28]=1
    if df.iloc[i,27] == 'Veuillez juste revoir vos étape et voir leurs conforme avec les norms du projet.':
        df.iloc[i,28]=2
    if df.iloc[i,27] == 'Veuillez vous revoir la clarté sur le cahier de charge':
        df.iloc[i,28]=3

# Update DataFrame
df.to_csv("DataTotal1.csv")



#---------------------------------------------------------------------------------|
#-----------------------------TF-IDF and KMeans-----------------------------------|
#---------------------------------------------------------------------------------| 

import pandas as pd

df = pd.read_csv("DataTotal1.csv")

R = df["remarque"]

# Fonction de netoyage
R = R.str.lower()
R = [x for x in R]

import re
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
def Netoyer(X): 
    X1 = []
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()   
       # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        X1.append(document)
        
    return X1

R = Netoyer(R)

# Make all words in lower format
L_Terms = []
for i in range(len(R)):
    L_Terms.append(R[i].lower().split())
    
# Liste of stop words
from nltk.corpus import stopwords
en_arrêt = set(stopwords.words('french'))
en_arrêt.update(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
en_arrêt.update(['1','2','3','4','5','6','7','8','9','0'])

from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()

# take off the stop words from list
BagWord = []
for j in L_Terms:
    # remove stop words from tokens
    tokens_arrêt = [i for i in j if not i in en_arrêt]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in tokens_arrêt]
    # ajouter tokens dans la liste
    BagWord.append(stemmed_tokens) 

#--------------------------------TF-IDF example on Python-------------------
    
# bag of word corrected    
BagWordC = []    
for i in range(len(BagWord)):
    BagWordMC = []
    for j in range(len(BagWord[i])):
        if(len(BagWord[i][j])>3):
            BagWordMC.append(BagWord[i][j])
    BagWordC.append(BagWordMC)
 



    
# Convert BagWordC into Sentence

Sentences = []
for i in range(len(BagWordC)):
    t = ' '.join(BagWordC[i])
    Sentences.append(t)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(Sentences)
kmeans = KMeans(n_clusters=3200).fit(tfidf)
clusters = kmeans.labels_.tolist()
df['Remarques_Clusters'] = clusters









