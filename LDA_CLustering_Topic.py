# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:12:15 2020
 LDA
@author: ABDEDDAIM Omar
"""
import requests
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer = SnowballStemmer("french")

# le lien pour consulter les données
URL="Your URl"
  
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
df['objectData.data.donnees.remarque'] = df['objectData.data.donnees.remarque'].str.lower() 
# removing everything except alphabets`
df['clean_doc'] = df['objectData.data.donnees.remarque'].str.replace("[^a-zA-Z#]", " ")
  
# Etape 2:  Preprocessus des données
 
X = list(df['clean_doc'])
X = str(X).strip('[]') # convertir liste à une chaine des caractères
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result

X = preprocess(X)


processed_docs = []

for doc in X:
    processed_docs.append(preprocess(doc))

# Step 3: Prendre un packet des mots dans la donnée


'''
Créer un dictionnaire à partir de 'processed_docs' contient le nombre des fois un mot a été mentioné
 dans  le groupe d'entrainement utilisant gensim.corpora.Dictionary  et nommé 'dictionary'
'''
dictionary = gensim.corpora.Dictionary(processed_docs)

'''
vérification que le  dictionaire est crée
'''
count = 0

for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# Gensim filter_extremes
'''
eliminer tous les mots les plus rare et les mots les plus commun:

- Mot apparait moins que 10 fois
- mots appraient plus que  10% de tout les mots dans le document 
'''
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)
    
# Gensim doc2bow pour convertir le dictionnaire à une liste des mots 


'''
créer un packet des mots (liste) i.e pour chaque document on crée un dictionnaire et un rapport
indique les mots et le nbr des fois qui appraissnet. Stock dans 'bow_corpus'
'''
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Modeling LDA with BOW

'''
Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
'''
# TODO
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 8, 
                                   id2word = dictionary,                                    
                                   passes = 1500,
                                   workers = 2)
'''
For each topic, we will explore the words occuring in that topic and its relative weight
'''
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

#: Tester model sur un document non vu 

URL_test="https://urbarokhas.karaz.org/karazortal/access/rest/kdata/search/referentiel_aviscommission_search_AllAvisCommission?loadXml=true&apiKey=AB90G-BH903-W4EE1-Z66Q9-7822K&offset=200&limit=400&sortInfo=id=ASC"
  
r_test = requests.get(URL_test) # envoie de demande à travers le lien URL
dictr_test = r_test.json() # importe les données format json
recs_test = dictr_test['data'] # Prendre que les données qui nous concerne dans le tableau. 
df_test = json_normalize(recs_test) # Rendre un json sous format dataframe 

 # Stocker la colonne des remarques--------------------------------
df_test = pd.DataFrame(df_test,columns=['objectData.data.donnees.remarque'])
df_test= df_test[df_test['objectData.data.donnees.remarque'] != '']# delete spaces
df_test= df_test[df_test['objectData.data.donnees.remarque'] != '.....']# delete points
df_test= df_test[df_test['objectData.data.donnees.remarque'] != '????']# delete points
df_test= df_test[df_test['objectData.data.donnees.remarque'].map(len) > 6]
# removing everything except alphabets`
df_test['clean_doc'] = df_test['objectData.data.donnees.remarque'].str.replace("[^a-zA-Z#]", " ")
  
# Etape 2:  Preprocessus des données
 
X_test = list(df_test['clean_doc'])
X_test = str(X_test).strip('[]')

processed_docs_test = []

for doc in X:
    processed_docs_test.append(preprocess(doc))

dictionary_test = gensim.corpora.Dictionary(processed_docs_test)

count = 0

for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

# Data preprocessing step for the unseen document
bow_vector = dictionary.doc2bow(preprocess(X_test))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
