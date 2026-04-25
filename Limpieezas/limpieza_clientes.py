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

ruta_clientes=r"C:\Users\angel\OneDrive\Escritorio\Datathon_Hey_2026_dataset_transacciones 1\dataset_transacciones\hey_clientes.csv"

clientes_raw=pd.read_csv(ruta_clientes)

print("clientes original:",clientes_raw.shape)

clientes=clientes_raw.copy()

clientes.columns=[c.lower() for c in clientes.columns]

before=len(clientes)

clientes=clientes.dropna(subset=["user_id"])
after_na=len(clientes)

clientes=clientes.drop_duplicates(subset=["user_id"])
after_dup=len(clientes)

print("after dropna:",after_na,"(-",before-after_na,")")
print("after duplicates:",after_dup,"(-",after_na-after_dup,")")
print("final clientes:",clientes.shape)

clientes.to_csv(r"C:\Users\angel\OneDrive\Escritorio\clientes_clean.csv",index=False)