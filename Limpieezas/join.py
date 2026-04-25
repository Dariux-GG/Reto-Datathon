import pandas as pd
import numpy as np
import unicodedata

convs=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\convs_clean.csv",encoding="latin1")
clientes=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\clientes_clean.csv")
productos=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\productos_clean.csv")
trans=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\trans_clean.csv")

conv_user=convs.groupby("user_id").agg({
"conv_id":"nunique",
"input":"count"
}).reset_index()

conv_user.columns=["user_id","num_convs","num_msgs"]


trans_user=trans.groupby("user_id").agg({
"transaccion_id":"count",
"monto":"sum"
}).reset_index()

trans_user.columns=["user_id","num_trans","monto_total"]


prod_user=productos.groupby("user_id").agg({
"producto_id":"count"
}).reset_index()

prod_user.columns=["user_id","num_productos"]


df=clientes \
.merge(conv_user,on="user_id",how="left") \
.merge(trans_user,on="user_id",how="left") \
.merge(prod_user,on="user_id",how="left")

print(df.shape)
print(df.head())

df=df.fillna(0)
print("users finales:",df["user_id"].nunique())
print(df.describe())

df.to_csv(r"C:\Users\angel\OneDrive\Escritorio\test.csv")