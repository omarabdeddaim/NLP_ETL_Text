# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 08:50:13 2020
c'est un alogrithm qui base sur les clusters utilisés dans K-means et les deux espaces de Tangent
Minanifold pour visualiser les l'impact des clutster sur chaque topic. 
 
@author: ABDEDDAIM Omar
"""
import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.io.json import json_normalize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
porter=PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
import umap
from sklearn.cluster import KMeans

# le lien pour consulter les données
URL="your Url"
  
r = requests.get(URL) # envoie de demande à travers le lien URL
dictr = r.json() # importe les données format json
recs = dictr['data'] # Prendre que les données qui nous concerne dans le tableau. 
df = json_normalize(recs) # Rendre un json sous format dataframe 

 # Stocker la colonne des remarques--------------------------------
df = pd.DataFrame(df,columns=['objectData.data.donnees.remarque'])
df= df[df['objectData.data.donnees.remarque'] != '']# delete spaces
df= df[df['objectData.data.donnees.remarque'] != '.....']# delete points
df= df[df['objectData.data.donnees.remarque'] != '????']# delete points
df= df[df['objectData.data.donnees.remarque'].map(len) > 6]
# removing everything except alphabets`
df['clean_doc'] = df['objectData.data.donnees.remarque'].str.replace("[^a-zA-Z#]", " ")

# removing short words
df['clean_doc'] = df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# make all text lowercase
df['clean_doc'] = df['clean_doc'].apply(lambda x: x.lower())

stop_words = stopwords.words('french')

# tokenization
tokenized_doc = df['clean_doc'].apply(lambda x: x.split())

# remove stop-words
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
tokenized_doc = [[' '.join(i)] for i in tokenized_doc]
X2 = []

#(flatten the list) faire la liste des listes en une liste
for x in tokenized_doc:
    for y in x:
        X2.append(y)
tokenized_doc = X2
# de-tokenization
detokenized_doc = []
for i in range(len(df)):
    t = ''.join(tokenized_doc[i])
    detokenized_doc.append(t)

# -------------------------------Stemming-------------
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
Liste_Stem = []
for line in detokenized_doc:
    stem_sentence=stemSentence(line)
    Liste_Stem.append(stem_sentence)

df['clean_doc'] = Liste_Stem



vectorizer = TfidfVectorizer(stop_words=stopwords.words('french'), 
max_features= 1000, # keep top 1000 terms 
max_df = 0.5, 
smooth_idf=True)

X = vectorizer.fit_transform(df['clean_doc'])

X.shape # check shape of the document-term matrix

#  modeling 
num_clusters = 7
km = KMeans(n_clusters=num_clusters)
km.fit(X)
clusters = km.labels_.tolist()
U, Sigma, VT = randomized_svd(X, n_components=10, n_iter=100,
 random_state=122)
"""
for i, comp in enumerate(VT):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    print("Concept "+str(i)+": ")
    for t in sorted_terms:
        print(t[0])
        print(" ")
"""
X_topics=U*Sigma
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=42).fit_transform(X_topics)

plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], 
 c = clusters,
 s = 10, # size
 edgecolor='none'
 )
plt.show()
