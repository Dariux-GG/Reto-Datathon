import pandas as pd
import numpy as np
import unicodedata

convs=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\convs_clean.csv",encoding="latin1")
clientes=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\clientes_clean.csv")
productos=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\productos_clean.csv")
trans=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\trans_clean.csv")


print(convs["user_id"].nunique())
print(clientes["user_id"].nunique())
print(productos["user_id"].nunique())
print(trans["user_id"].nunique())
