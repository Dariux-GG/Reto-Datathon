import pandas as pd
import numpy as np
import unicodedata

convs=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\Datathon_Hey_dataset_conversaciones 1\dataset_conversaciones\dataset_50k_anonymized.csv",encoding="latin1")
clientes=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\Datathon_Hey_2026_dataset_transacciones 1\dataset_transacciones\hey_clientes.csv")
productos=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\Datathon_Hey_2026_dataset_transacciones 1\dataset_transacciones\hey_productos.csv")
trans=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\Datathon_Hey_2026_dataset_transacciones 1\dataset_transacciones\hey_transacciones.csv")


print(convs["user_id"].nunique())
print(clientes["user_id"].nunique())
print(trans["user_id"].nunique())