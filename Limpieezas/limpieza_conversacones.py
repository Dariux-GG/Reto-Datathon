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

#Ruta
ruta_convs=r"C:\Users\angel\OneDrive\Escritorio\Datathon_Hey_dataset_conversaciones 1\dataset_conversaciones\dataset_50k_anonymized.csv"


convs_raw=pd.read_csv(ruta_convs,encoding="latin1")

print("convs original:",convs_raw.shape)

convs=convs_raw.copy()

convs.columns=[c.lower() for c in convs.columns]

convs["input"]=convs["input"].apply(normalize_text)
convs["output"]=convs["output"].apply(normalize_text)

convs["date"]=pd.to_datetime(convs["date"],errors="coerce")

convs["channel_source"]=convs["channel_source"].astype(str)

before_drop=len(convs)

convs=convs.dropna(subset=["user_id","conv_id"])
after_na=len(convs)

convs=convs.drop_duplicates()
after_dup=len(convs)

print("after dropna:",after_na,"(-",before_drop-after_na,")")
print("after duplicates:",after_dup,"(-",after_na-after_dup,")")
print("final convs:",convs.shape)

convs.to_csv(r"C:\Users\angel\OneDrive\Escritorio\convs_clean.csv",index=False)