import pandas as pd
pd.set_option("display.max_columns",None)
pd.set_option("display.width",None)
import numpy as np
import unicodedata

convs=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\convs_clean.csv",encoding="latin1")
clientes=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\clientes_clean.csv")
productos=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\productos_clean.csv")
trans=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\trans_clean.csv")
df=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\test.csv")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X=df[["ingreso_mensual_mxn","score_buro","num_productos_activos","num_trans","monto_total","dias_desde_ultimo_login","satisfaccion_1_10"]]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

kmeans=KMeans(n_clusters=5,random_state=42)
df["cluster"]=kmeans.fit_predict(X_scaled)
a=df.groupby("cluster").mean(numeric_only=True)
print(a)
b=df.groupby("cluster").size()
print(b)