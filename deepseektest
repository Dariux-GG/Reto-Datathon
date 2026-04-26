import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests
import json
from dotenv import load_dotenv
import os
load_dotenv()
MI_CLAVE_DEEPSEEK=os.getenv("DEEPSEEK_API_KEY")


def deepseek_chat(messages,api_key,modelo="deepseek-v4-flash",temp=0.7):
    url_api="https://api.deepseek.com/v1/chat/completions"
    
    cuerpo_json={
        "model":modelo,
        "messages":messages,
        "temperature":temp,
        "stream":False
    }
    
    headers={
        "Content-Type":"application/json",
        "Authorization":f"Bearer {api_key}"
    }
    
    respuesta=requests.post(url_api,headers=headers,data=json.dumps(cuerpo_json))
    
    if not respuesta.ok:
        raise RuntimeError(respuesta.text)
    
    contenido=respuesta.json()
    return contenido["choices"][0]["message"]["content"]


df=pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\test.csv")

cols=["ingreso_mensual_mxn","score_buro","num_productos_activos","num_trans","monto_total","dias_desde_ultimo_login","satisfaccion_1_10"]

X=df[cols]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

kmeans=KMeans(n_clusters=5,random_state=42)
df["cluster"]=kmeans.fit_predict(X_scaled)



def estrategia_cluster(c):
    if c==0:
        return "Es un cliente premium, sugiere inversiones."
    if c==1:
        return "Cliente con bajo score, evita créditos agresivos."
    if c==4:
        return "Usuario inactivo, enfócate en reactivación."
    return "Ofrece recomendaciones generales."


def construir_contexto(user_row):
    estrategia=estrategia_cluster(user_row["cluster"])
    
    return f"""
Eres Havi, el asistente de Hey Banco.

Información del usuario:
- Edad: {user_row["edad"]}
- Ingreso mensual: ${user_row["ingreso_mensual_mxn"]:,.0f}
- Score crediticio: {user_row["score_buro"]}
- Productos activos: {user_row["num_productos_activos"]}
- Número de transacciones: {user_row["num_trans"]}
- Monto total movido: ${user_row["monto_total"]:,.0f}
- Satisfacción: {user_row["satisfaccion_1_10"]}
- Cluster: {user_row["cluster"]}

Estrategia:
{estrategia}

Instrucciones:
- Responde de forma natural y útil
- Personaliza la respuesta según el perfil
- Da recomendaciones financieras accionables
"""


#usuario
user_id_demo="USR-13688"
user_filtered=df[df["user_id"]==user_id_demo]

if user_filtered.empty:
    raise ValueError("Usuario no encontrado")

user_data=user_filtered.iloc[0]


#loop de chat
historial=[]

# construir contexto
contexto=construir_contexto(user_data)
historial.append({"role":"system","content":contexto})

while True:
    mensaje_usuario=input("\nTú: ")
    
    if mensaje_usuario.lower()=="salir":
        break
    
    historial.append({"role":"user","content":mensaje_usuario})
    
    respuesta=deepseek_chat(
        messages=historial,
        api_key=MI_CLAVE_DEEPSEEK
    )
    
    historial.append({"role":"assistant","content":respuesta})
    
    print("\nHavi:",respuesta)