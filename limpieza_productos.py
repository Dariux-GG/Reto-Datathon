import pandas as pd
import numpy as np
import unicodedata

def fix_encoding(text):
    if isinstance(text,str):
        try:
            return text.encode("latin1").decode("utf-8")
        except:
            return text
    return text

def normalize_text(text):
    if isinstance(text,str):
        text=fix_encoding(text)
        text=unicodedata.normalize("NFKC",text)
        return text.strip().lower()
    return text


ruta_productos=r"C:\Users\angel\OneDrive\Escritorio\Datathon_Hey_2026_dataset_transacciones 1\dataset_transacciones\hey_productos.csv"


productos_raw=pd.read_csv(ruta_productos)

print("productos original:",productos_raw.shape)

productos=productos_raw.copy()

productos.columns=[c.lower() for c in productos.columns]

before=len(productos)

productos=productos.dropna(subset=["producto_id","user_id"])
after_na=len(productos)

productos=productos.drop_duplicates(subset=["producto_id"])
after_dup=len(productos)

print("after dropna:",after_na,"(-",before-after_na,")")
print("after duplicates:",after_dup,"(-",after_na-after_dup,")")
print("final productos:",productos.shape)

productos.to_csv(r"C:\Users\angel\OneDrive\Escritorio\productos_clean.csv",index=False)