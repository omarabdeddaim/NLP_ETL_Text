# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:05:30 2020

@author: SURFACE Pro 5
"""
import pandas as pd


df = pd.read_csv("DataFinish.csv")
df=df.dropna()
df= df[df['Rem_ACT'].str.lower()]
df= df[df['Rem_ACT'] != 'observations']
df= df[df['Rem_ACT'] != 'so']
#favorable, Avis favorable, Favorable,Sans objection. , Sans objection, sans objection.
#
df= df[df['Rem_ACT'] != 'sans objection.']


X = df['Rem_ACT']
X = X.str.lower()
"""
X = X[~X.str.contains("observations", na=False)]
X = X[~X.str.contains("so", na=False)]
X = X[~X.str.contains("favorable", na=False)]
X = X[~X.str.contains("sans objection", na=False)]
"""
X = [x for x in X]

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

X = Netoyer(X)

# Make all words in lower format
L_Terms = []
for i in range(len(X)):
    L_Terms.append(X[i].lower().split())
    
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
 
# Calculating bag of words put it in set
word_set = set()
for i in range(len(BagWordC)):
    word_set.update(set(BagWordC[i]))

#Stock the dictionnaire for each row and affecte the current value
word_dict = []

for i in range(len(BagWordC)):
     T = dict.fromkeys(word_set, 0)
     for word in BagWordC[i]:
         T[word] += 1    
     word_dict.append(T)    
     print(i)
"""        
 #----------------------------------term frequency--------------------------
       
def compute_tf(word_dict, l):
    tf = {}
    sum_nk = len(l)
    for word, count in word_dict.items():
        tf[word] = count/sum_nk
    return tf  
    
TF = []   
for i in range(len(word_dict)):
    TF.append(compute_tf(word_dict[i], BagWordC[i]))   

#-----------------------------------Term Inverse DF----------------------
import math
def compute_idf(strings_list):
    n = len(strings_list)
    idf = dict.fromkeys(strings_list[0].keys(), 0)
    for l in strings_list:
        for word, count in l.items():
            if count > 0:
                idf[word] += 1
    
    for word, v in idf.items():
        idf[word] = math.log(n / float(v))
    return idf     
        
IDF =compute_idf([ t for t in word_dict])

def compute_tf_idf(tf, idf):
    tf_idf = dict.fromkeys(tf.keys(), 0)
    for word, v in tf.items():
        tf_idf[word] = v * idf[word]
    return tf_idf

tf_idf =[]
for i in range(len(BagWordC)):
    tf_idf.append(compute_tf_idf(TF[i], IDF))
"""

"""
/////////////////////////////////////
Application TF-IDF avec K-Means (utilisation de la bibliothéque prête)
                                    //////////////////////////////////////////////
"""


# Convert BagWordC into Sentence

Sentences = []
for i in range(len(BagWordC)):
    t = ' '.join(BagWordC[i])
    Sentences.append(t)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(Sentences)
kmeans = KMeans(n_clusters=32).fit(tfidf)
clusters = kmeans.labels_.tolist()

df['remarques'] = clusters

"""
lines_for_predicting = ["fournir ministèr tutel", 
                        "fournir ministèr tutel fournir contrat architect viser ordr architect fournir certificat propriété date récent"]


kmeans.predict(tfidf_vectorizer.transform(lines_for_predicting))
"""
"""

df.to_csv(r'Data1.csv')


import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
es = Elasticsearch(http_compress=True)
        
df = pd.read_csv("Data1.csv")
df=df.drop(columns=['Unnamed: 0.1'])
Nb = [t for t in range(len(df))]
df['inc'] = Nb
use_these_keys = ["inc","DossierId","reference","ref_ACT","cout","Cout_ACT","CatgeProjet",
                  "Pjt_ACT","MemCom","MeCo_ACT","ArchID","ArchID_ACT","remarques","Rem_ACT"]

def filterKeys(document):
    return {key: document[key] for key in use_these_keys }


      
def doc_generator(df):
    df_iter = df.iterrows()
    for index, document in df_iter:
        yield {
                "_index": 'rokhas1',
                "_type": "_doc",
                "_id" : f"{document['inc']}",
                "_source": filterKeys(document),
            } 

    #raise StopIteration 


helpers.bulk(es, doc_generator(df))


X = [t for t in df]

from es_pandas import es_pandas

################################################################
# Information of es cluseter
es_host = 'localhost:9200'
index = 'rokhas1'

# crete es_pandas instance
ep = es_pandas(es_host)

# Example of read data from es
dataset = ep.to_pandas(index)
dataset = dataset.drop(columns=['inc'])
"""


        
        
        
        
        
