# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:39:34 2020

@author: SURFACE Pro 5
"""


import codecs
import re
from collections import Counter
import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
import matplotlib.pyplot as plt
import pandas as pd
import json
import xmltodict
from pandas.io.json import json_normalize
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
X = dataset.iloc[:, [0, 2, 4, 6, 9, 11]].values
y = dataset.iloc[:, 12].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
clf = LinearSVC(C=10, penalty="l2", dual=False, tol=1e-3).fit(X_train, y_train)


# Distance de Levenhstein
# Essay 2


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


# Function Compiler

def compiler(A):
    Bo1 = False
    L1 = []
    Bo3 = False
    L3 = []
    Bo4 = False
    L4 = []
    Bo5 = False
    L5 = []
    for i in range(len(dataset)):
        if(dataset.iloc[i, 10] == A[1]):
            a1 = dataset.iloc[i, 11]
            Bo1 = True
    if(Bo1 == False):
        for i in range(len(dataset)):
            L1.append(lev_dist(A[1], dataset.iloc[i, 10]))
        for i in range(len(dataset)):
            if(L1[i] == min(L1)):
                a1 = dataset.iloc[i, 11]

    for i in range(len(dataset)):
        if(dataset.iloc[i, 7] == A[3]):
            a3 = dataset.iloc[i, 2]
            bo3 = True
    if(Bo3 == False):
        for i in range(len(dataset)):
            L3.append(lev_dist(A[3], dataset.iloc[i, 7]))
        for i in range(len(dataset)):
            if(L3[i] == min(L3)):
                a3 = dataset.iloc[i, 2]

    for i in range(len(dataset)):
        if(dataset.iloc[i, 5] == A[4]):
            a4 = dataset.iloc[i, 6]
            bo4 = True
    if(Bo4 == False):
        for i in range(len(dataset)):
            L4.append(lev_dist(A[4], dataset.iloc[i, 5]))
        for i in range(len(dataset)):
            if(L4[i] == min(L4)):
                a4 = dataset.iloc[i, 6]

    for i in range(len(dataset)):
        if(dataset.iloc[i, 1] == A[5]):
            a5 = dataset.iloc[i, 0]
    if(Bo5 == False):
        for i in range(len(dataset)):
            L5.append(lev_dist(A[5], dataset.iloc[i, 1]))
        for i in range(len(dataset)):
            if(L5[i] == min(L5)):
                a5 = dataset.iloc[i, 0]

    K = [int(A[0]), int(a1), int(A[2]), int(a3), int(a4), int(a5)]
    return K

################################################


def predict(data):
    A = compiler(data)
    A = np.array(A).reshape(1, -1)
    A = np.append(A, clf.predict(A))
    A = A.reshape(1, -1)
    liste = []
    for i in range(len(dataset)):
        if(dataset.iloc[i, 12] == A[0][6]):
            liste.append(dataset.iloc[i, 8])
    LISTE = Elimination(liste)
    return liste

{
  "DossierID": 213001,
  "reference": "PCT-PMHD-AHD-111/2017",
  "Cout_ACT": 0,
  "Pjt_ACT": "GPJ",
  "MeCo_ACT": "PROTECTION CIVILE MOHAMMEDIA",
  "ArchID_ACT": "AmMo.Mekouar@casaurba"
}

A = [213143,"PCT-PNCR-BSK-142/2017",0,"GPJ","PROTECTION CIVILE NOUACER","Ah.Maktoum@casaurba"]
L = predict(A)

tokenize = lambda doc: doc.lower().split(" ")
tokenized_documents= [tokenize(d) for d in L]

Jacand = []
a = 0
b = 0
for i in range(len(L)):
    a = len(set(tokenized_documents[i]).intersection(set(tokenized_documents[6-i])))
    b = len(set(tokenized_documents[i]).union(set(tokenized_documents[6-i])))
    Jacand.append(a/b)


tokenize = lambda doc: doc.lower().split(" ")
def jacanda(L,T):
    a = len(set(set(tokenize(L))).intersection(set(tokenize(T))))
    b = len(set(set(tokenize(L))).union(set(tokenize(T))))
    Jacand = a/b
    return Jacand  

def Elimination(L):
    LISTEA = []
    Bo1 = False 
    Bo2 = False
    Bo3 = False 
    Bo4 = False 
    Bo5 = False 
    Bo6 = False
    Liste0 = [L[0]]
    for i in range(len(L)):
        if(jacanda(L[0],L[i]) < 0.5):
            Liste0.append(L[i])
    
    Liste1 = []
    if(len(Liste0)>3):
        Bo1 = True
        Liste1.append(Liste0[0])
        Liste1.append(Liste0[1])
        for i in range(2,len(Liste0)):
            if(jacanda(Liste0[1],Liste0[i])<0.3):
                Liste1.append(Liste0[i])
    Liste2 = []
    if(len(Liste1)>4):
        Bo2 = True
        Liste2.append(Liste1[0])
        Liste2.append(Liste1[1])
        Liste2.append(Liste1[2])
        for i in range(3,len(Liste1)):
            if(jacanda(Liste1[2],Liste1[i])<0.3):
                Liste2.append(Liste1[i])  
    Liste3 = []
    if(len(Liste2)>5):
        Bo3 = True
        Liste3.append(Liste2[0])
        Liste3.append(Liste2[1])
        Liste3.append(Liste2[2])
        Liste3.append(Liste2[3])
        for i in range(4,len(Liste2)):
            if(jacanda(Liste2[2],Liste2[i])<0.3):
                Liste3.append(Liste2[i])    
    Liste4 = []
    if(len(Liste3)>6):
        Bo4 = True
        Liste4.append(Liste3[0])
        Liste4.append(Liste3[1])
        Liste4.append(Liste3[2])
        Liste4.append(Liste3[3])
        Liste4.append(Liste3[4])
        for i in range(5,len(Liste3)):
            if(jacanda(Liste3[2],Liste3[i])<0.3):
                Liste4.append(Liste3[i])
    Liste5 = []
    if(len(Liste4)>7):
        Bo5 = True
        Liste5.append(Liste4[0])
        Liste5.append(Liste4[1])
        Liste5.append(Liste4[2])
        Liste5.append(Liste4[3])
        Liste5.append(Liste4[4])
        Liste5.append(Liste4[5])
        for i in range(6,len(Liste4)):
            if(jacanda(Liste4[2],Liste4[i])<0.3):
                Liste5.append(Liste4[i])
    Liste6 = []
    if(len(Liste5)>8):
        Bo6 = True
        Liste6.append(Liste5[0])
        Liste6.append(Liste5[1])
        Liste6.append(Liste5[2])
        Liste6.append(Liste5[3])
        Liste6.append(Liste5[4])
        Liste6.append(Liste5[5])
        for i in range(7,len(Liste6)):
            if(jacanda(Liste5[2],Liste5[i])<0.3):
                Liste6.append(Liste5[i])
                
    if(Bo1==True and Bo2==False and Bo3==False and Bo4==False and Bo5==False and Bo6==False):
        LISTEA = Liste1
    if(Bo1 ==True and Bo2==True and Bo3==False and Bo4==False and Bo5==False and Bo6==False):
        LISTEA = Liste2
    if(Bo1 ==True and Bo2==True and Bo3==True and Bo4==False and Bo5==False and Bo6==False):
        LISTEA = Liste3
    if(Bo1 ==True and Bo2==True and Bo3==True and Bo4==True and Bo5==False and Bo6==False):
        LISTEA = Liste4  
    if(Bo1 ==True and Bo2==True and Bo3==True and Bo4==True and Bo5==True and Bo6==False):
        LISTEA = Liste5
    if(Bo1 ==True and Bo2==True and Bo3==True and Bo4==True and Bo5==True and Bo6==True):
        LISTEA = Liste6 
    
    return LISTEA
LISTE = Elimination(L)     
            for j in range(len(L)-i):
                if(jacanda(L[i],L[j])<0.5):
                    Liste.append(L[j])
                    BO =True
     
       
from collections import OrderedDict


L1 = list(OrderedDict.fromkeys(S.split("\n")))       
        

# Convert liste into string

S = str('\n'.join((str(i) for i in L))) 
        



