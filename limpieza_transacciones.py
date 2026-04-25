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


ruta_trans=r"C:\Users\angel\OneDrive\Escritorio\Datathon_Hey_2026_dataset_transacciones 1\dataset_transacciones\hey_transacciones.csv"

trans_raw=pd.read_csv(ruta_trans)

print("trans original:",trans_raw.shape)

trans=trans_raw.copy()

trans.columns=[c.lower() for c in trans.columns]

before=len(trans)

trans=trans.dropna(subset=["transaccion_id","user_id"])
after_na=len(trans)

trans=trans.drop_duplicates(subset=["transaccion_id"])
after_dup=len(trans)

print("after dropna:",after_na,"(-",before-after_na,")")
print("after duplicates:",after_dup,"(-",after_na-after_dup,")")
print("final trans:",trans.shape)