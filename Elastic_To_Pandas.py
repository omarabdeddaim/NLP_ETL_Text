import time

import pandas as pd

from es_pandas import es_pandas

# Information of es cluseter
es_host = 'localhost:9200'
index = 'rokhas'

# crete es_pandas instance
ep = es_pandas(es_host)

# Example of read data from es
df = ep.to_pandas(index)
print(df.head())
