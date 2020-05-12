# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:54:06 2020

@author: Omar abdeddaim
"""



import requests
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import re
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
# Apporter les données de tout le dossier et les traiter d'une façon générale 
import matplotlib.pyplot as plt

"""
###########################################################################################
Collecte Data from Dossier
##########################################################################################
"""
df2 = pd.DataFrame()
df= pd.DataFrame()
L=[]
T=0
# Importer les données
while(T<=100):
    URL = "Your Url"+str(85800+100*T)+"&limit=100&sortInfo=id=ASC&loadXml=true"
    r = requests.get(URL) # envoie de demande à travers le lien URL
    print(T)
    if r.status_code == 200:
        try:
            dictr = r.json() # importe les données format json
            recs = dictr['data'] # Prendre que les données qui nous concerne dans le tableau. 
            df = json_normalize(recs)
            df1 = pd.DataFrame(df,columns=['id',
                                   'objectData.data.dataII.histoCommissions.existeAutreCom',
                                   'objectData.data.dataII.travaux.cout',
                                   'objectData.data.donnees.archivage.autorisation.fileSize',# but d'en trouver le volume moyenne
                                   'objectData.data.donnees.archivage.plan.fileSize',# Comparer en étudiant avec les remarques
                                   'objectData.data.donnees.decisions.president.decision',
                                   'objectData.data.donnees.decisions.president.motif',
                                   'objectData.data.donnees.guichet.depot.description',
                                   'objectData.data.donnees.infosGenerales.allMaitresOuvrages',
                                   'objectData.data.donnees.infosGenerales.arrondissement.description',
                                   'objectData.data.donnees.infosGenerales.categorieProjet',
                                   'objectData.data.donnees.infosGenerales.codeArrondissement',
                                   'objectData.data.donnees.infosGenerales.codePrefecture',
                                   'objectData.data.donnees.infosGenerales.consistance',                        
                                   'objectData.data.donnees.infosGenerales.nature',
                                   'objectData.data.donnees.infosGenerales.nombreNiveaux',
                                   'objectData.data.donnees.infosGenerales.qualiteMO',
                                   'objectData.data.donnees.infosGenerales.terrain.adresseDetailles.ville.description',
                                   'objectData.data.donnees.infosGenerales.terrain.surfaceTerrain',
                                   'objectData.data.donnees.infosGenerales.type',
                                   'objectData.data.donnees.membresCommission.secGeneral.decision',
                                   'objectData.data.donnees.membresCommission.secGeneral.motif',
                                   'objectData.data.donnees.reintroduction.decision',
                                   'objectData.data.donnees.reintroduction.note',
                                   'objectData.data.donnees.rejetDepot.historique.dep.dateTraitement',
                                   'objectData.data.donnees.rejetDepot.historique.dep.motif',
                                   'objectData.data.donnees.rejetDepot.motif',
                                   'objectData.data.donnees.traitement.dateCreation',
                                   'objectData.data.donnees.traitement.dateModification',
                                   'objectData.data.nbrAjournemntGL',
                                   'objectData.data.nbrBackTocom',
                                   'objectData.data.nbrReintroduction',
                                   'objectData.data.nbrRejetDossier',
                                   'objectData.data.nbrUpdatedoc',
                                   'objectData.data.paiement.amount',
                                   'objectData.data.paiement.amountMAD',
                                   'objectData.data.paiement.mode',
                                   'objectData.data.paiement.payConnect.message',
                                   'objectData.data.reference',
                                   'objectData.data.step',
                                   'objectData.data.to',
                                   'objectData.data.verssion',
                                   'objectData.data.donnees.reintroduction.historique.reintro.dateTraitement',
                                   'objectData.data.donnees.reintroduction.historique.reintro.decision',
                                   'objectData.data.donnees.reintroduction.historique.reintro.note',
                                   
                               ])
            df2 = df2.append(df1)
            df2.reset_index(drop=True, inplace=True)  
            print("Json")
            print(r.status_code)
            T +=1
        except ValueError as e:
            dictr = r.text # importe les données format text
            L.append(dictr)
            print("text")
            print(r.status_code)
            T+=1
    elif r.status_code == 404:
        T=900
        print(r.status_code)

# convert text into Json and stock it in test.txt   

"""
I validate the data which isn't json detected in the plateforme 

https://jsonlint.com/?code=

"""
        # text compiler à DataFrame
import json
import pandas as pd
with open('test.txt',encoding='utf8') as json_data:
    data = json.load(json_data)
    dftest = json_normalize(data) 
    
# Stocker 47 caractères dans une autre dataFrame    
dftest1 = pd.DataFrame(dftest,columns=['id',
                                   'objectData.data.dataII.histoCommissions.existeAutreCom',
                                   'objectData.data.dataII.travaux.cout',
                                   'objectData.data.donnees.archivage.autorisation.fileSize',# but d'en trouver le volume moyenne
                                   'objectData.data.donnees.archivage.plan.fileSize',# Comparer en étudiant avec les remarques
                                   'objectData.data.donnees.decisions.president.decision',
                                   'objectData.data.donnees.decisions.president.motif',
                                   'objectData.data.donnees.guichet.depot.description',
                                   'objectData.data.donnees.infosGenerales.allMaitresOuvrages',
                                   'objectData.data.donnees.infosGenerales.arrondissement.description',
                                   'objectData.data.donnees.infosGenerales.categorieProjet',
                                   'objectData.data.donnees.infosGenerales.codeArrondissement',
                                   'objectData.data.donnees.infosGenerales.codePrefecture',
                                   'objectData.data.donnees.infosGenerales.consistance',                        
                                   'objectData.data.donnees.infosGenerales.nature',
                                   'objectData.data.donnees.infosGenerales.nombreNiveaux',
                                   'objectData.data.donnees.infosGenerales.qualiteMO',
                                   'objectData.data.donnees.infosGenerales.terrain.adresseDetailles.ville.description',
                                   'objectData.data.donnees.infosGenerales.terrain.surfaceTerrain',
                                   'objectData.data.donnees.infosGenerales.type',
                                   'objectData.data.donnees.membresCommission.secGeneral.decision',
                                   'objectData.data.donnees.membresCommission.secGeneral.motif',
                                   'objectData.data.donnees.reintroduction.decision',
                                   'objectData.data.donnees.reintroduction.note',
                                   'objectData.data.donnees.rejetDepot.historique.dep.dateTraitement',
                                   'objectData.data.donnees.rejetDepot.historique.dep.motif',
                                   'objectData.data.donnees.rejetDepot.motif',
                                   'objectData.data.donnees.traitement.dateCreation',
                                   'objectData.data.donnees.traitement.dateModification',
                                   'objectData.data.nbrAjournemntGL',
                                   'objectData.data.nbrBackTocom',
                                   'objectData.data.nbrReintroduction',
                                   'objectData.data.nbrRejetDossier',
                                   'objectData.data.nbrUpdatedoc',
                                   'objectData.data.paiement.amount',
                                   'objectData.data.paiement.amountMAD',
                                   'objectData.data.paiement.mode',
                                   'objectData.data.paiement.payConnect.message',
                                   'objectData.data.reference',
                                   'objectData.data.step',
                                   'objectData.data.to',
                                   'objectData.data.verssion',
                                   'objectData.data.donnees.reintroduction.historique.reintro.dateTraitement',
                                   'objectData.data.donnees.reintroduction.historique.reintro.decision',
                                   'objectData.data.donnees.reintroduction.historique.reintro.note',
                                   
                               ]) 

df2 = df2.append(dftest1)
df2.reset_index(drop=True, inplace=True)    
#convert dataFrame into CSV
df2.to_csv(r'Dossier1_20.csv')

#check Csv good set up
df = pd.read_csv("Dossier1_20.csv")


"""
####################################################################################################
collecte Data For Avis De Commission
################################################################################################
"""
import requests
import sys
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
dfC2 = pd.DataFrame()
dfC= pd.DataFrame()
L1 = []
T1 =0
######################offset = 55100
# Importer les données
while(T1<=500):
    URLC = "yourUrl"+str(55200+T1*100)+"&limit=100&sortInfo=id=ASC"  
    try:
        rC = requests.get(URLC) # envoie de demande à travers le lien URL
        print(T1)
        if rC.status_code == 200:
            try:
                dictrC = rC.json() # importe les données format json
                recsC = dictrC['data'] # Prendre que les données qui nous concerne dans le tableau. 
                dfC = json_normalize(recsC)
                dfC1 = pd.DataFrame(dfC,columns=['id',
                                'objectData.data.categorieProjet',
                                'objectData.data.donnees.decision',
                                'objectData.data.donnees.historique.decision',
                                'objectData.data.donnees.historique.remarque',
                                'objectData.data.donnees.remarque',
                                'objectData.data.donnees.validation.decision',
                                'stringIndex3',
                                'objectData.data.donnees.validation.historique.actions.dateTraitement',
                                'objectData.data.statut',
                                'objectData.data.to',
                                'objectData.data.traitement.dateCreation',
                                'objectData.data.traitement.dateModification',])
                dfC2 = dfC2.append(dfC1)
                dfC2.reset_index(drop=True, inplace=True)  
                print("Json")
                print(rC.status_code)
                T1 +=1
            except ValueError as e:
                dictrC = rC.text # importe les données format text
                L1.append(dictrC)
                print("text")
                print(rC.status_code)
                T1+=1
        elif rC.status_code == 404:
            T1=900
            print(rC.status_code)
    except requests.exceptions.RequestException as e:    # This is the correct syntax
        print (e)
        print(T1)
        T1 +=1

# convert text into Json and stock it in test.txt   


"""
I validate the data which isn't json detected in the plateforme 

https://jsonlint.com/?code=

"""
# text compiler à DataFrame
import json
import pandas as pd
with open('test.txt',encoding='utf8') as json_dataC:
    dataC = json.load(json_dataC)
    dftestC = json_normalize(dataC) 

dftestC1 = pd.DataFrame(dftestC,columns=['id',
                                'objectData.data.categorieProjet',
                                'objectData.data.donnees.decision',
                                'objectData.data.donnees.historique.decision',
                                'objectData.data.donnees.historique.remarque',
                                'objectData.data.donnees.remarque',
                                'objectData.data.donnees.validation.decision',
                                'stringIndex3',
                                'objectData.data.donnees.validation.historique.actions.dateTraitement',
                                'objectData.data.statut',
                                'objectData.data.to',
                                'objectData.data.traitement.dateCreation',
                                'objectData.data.traitement.dateModification',]) 

dfC2 = dfC2.append(dftestC1)
dfC2.reset_index(drop=True, inplace=True)    
#convert dataFrame into CSV
dfC2.to_csv(r'AVIS1_17.csv')

#check Csv good set up
dfC = pd.read_csv("AVIS14_15.csv")



"""
###############################################################################################
Connexion with ElasticSearch 
###############################################################################################    
"""
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
es = Elasticsearch(http_compress=True)
Nb = [t for t in range(4500)]
df1['inc'] = Nb
use_these_keys = ['inc','']


def filterKeys(document):
    return {key: document[key] for key in use_these_keys }              
def doc_generator(K):
    df_iter = K.iterrows()
    for index, document in df_iter:
        yield {
                "_index": 'fun',
                "_type": "_doc",
                "_id" : f"{document['inc']}",
                "_source": filterKeys(document),
            } 

    raise StopIteration 


helpers.bulk(es, doc_generator(df1),chunk_size=4500,request_timeout=60)       
        
   
  
                                  
#########################################################################################    
    












