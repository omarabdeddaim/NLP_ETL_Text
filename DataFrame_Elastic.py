import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
es = Elasticsearch(http_compress=True)
        
df = pd.read_csv("DataFinish.csv")
df=df.drop(columns=['Unnamed: 0'])
Nb = [t for t in range(568)]
df['inc'] = Nb
use_these_keys = ["inc","DossierId","reference","ref_ACT","cout","Cout_ACT","CatgeProjet",
                  "Pjt_ACT","MemCom","MeCo_ACT","ArchID","ArchID_ACT","remarques","Rem_ACT"]
def filterKeys(document):
    return {key: document[key] for key in use_these_keys }


      
def doc_generator(df):
    df_iter = df.iterrows()
    for index, document in df_iter:
        yield {
                "_index": 'rokhas',
                "_type": "_doc",
                "_id" : f"{document['inc']}",
                "_source": filterKeys(document),
            } 

    #raise StopIteration 


helpers.bulk(es, doc_generator(df))
