import pandas as pd


clientes=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\clientes_clean.csv") 
productos=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\productos_clean.csv") 
trans=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\trans_clean.csv") 
df=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\test.csv")

print(df.corr(numeric_only=True))
import matplotlib.pyplot as plt

df.hist(figsize=(12,10))
plt.show()
import numpy as np

df["log_monto_total"]=np.log1p(df["monto_total"])
df["log_num_msgs"]=np.log1p(df["num_msgs"])

cols=[
"edad",
"ingreso_mensual_mxn",
"score_buro",
"dias_desde_ultimo_login",
"num_trans",
"monto_total",
"num_productos"
]

X=df[cols]


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=4,random_state=42,n_init=10)
clusters=kmeans.fit_predict(X_scaled)

df["cluster"]=clusters

print(df.groupby("cluster")[cols].mean())



"""
Interpretación de clusters
Cluster 0 → High Value / Premium
ingreso: 48k 
score: 714 
monto_total: ~600k
transacciones: altas
activos 

upsell 
beneficios premium
retención 





Cluster 1 → Jóvenes hiperactivos
edad: 26
transacciones: 70 
login: muy frecuente
ingreso: medio-bajo
monto: medio

convertirlos a mayor valor (créditos, inversiones)
empujarlos a subir ticket promedio







Cluster 2 → Dormidos / Churn risk
dias sin login: 106 
transacciones: ~10 
monto: muy bajo (~42k)
pocos productos

campañas de reactivación
promos agresivas
si no regresan → no gastar más








Cluster 3 → Promedio / Masivo
ingreso: ~20k
comportamiento moderado en todo
ni destacan ni están muertos

nudges para subirlos a cluster 0 o 1
cross-sell
"""