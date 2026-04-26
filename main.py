from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import requests, json, os
from dotenv import load_dotenv

load_dotenv()
MI_CLAVE_DEEPSEEK = os.getenv("DEEPSEEK_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Setup al arrancar ---
df = pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\test.csv")
df_convs = pd.read_csv(r"C:\Users\angel\OneDrive\Escritorio\convs_clean.csv")
df_convs["input"] = df_convs["input"].astype(str).str.strip()
df_convs["output"] = df_convs["output"].astype(str).str.strip()
cols = ["edad","ingreso_mensual_mxn","score_buro","dias_desde_ultimo_login","num_trans","monto_total","num_productos"]
X = df[cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)



def cargar_historial_previo(user_id):
    convs = df_convs[df_convs["user_id"] == user_id].sort_values("date")
    mensajes = []
    for _, row in convs.iterrows():
        if row["input"] and row["input"] != "nan":
            mensajes.append({"role": "user", "content": row["input"]})
        if row["output"] and row["output"] != "nan":
            mensajes.append({"role": "assistant", "content": row["output"]})
    return mensajes
# Historial en memoria: { user_id: [mensajes] }
sesiones: dict = {}

# Helpers 
def estrategia_cluster(c):
    if c == 0: return "Cliente High Value/Premium. Ingreso alto (~48k), score 714, monto ~600k. Enfócate en upsell, beneficios premium y retención."
    if c == 1: return "Cliente joven hiperactivo (~26 años), muchas transacciones, ingreso medio-bajo. Empújalo a créditos e inversiones para subir ticket."
    if c == 2: return "Cliente dormido/churn risk. Sin login ~106 días, pocas transacciones. Campaña de reactivación agresiva o descartarlo."
    if c == 3: return "Cliente promedio/masivo. Comportamiento moderado en todo. Cross-sell y nudges para subirlo de cluster."
    return "Ofrece recomendaciones generales."

def construir_contexto(user_row):
    estrategia = estrategia_cluster(user_row["cluster"])
    
    return f"""
Eres Havi, asistente de Hey Banco.

PERFIL DEL USUARIO (uso interno, NO mencionar explícitamente salvo que sea relevante):
- Edad: {user_row["edad"]}
- Ingreso mensual: {user_row["ingreso_mensual_mxn"]}
- Score crediticio: {user_row["score_buro"]}
- Productos activos: {user_row["num_productos_activos"]}
- Número de transacciones: {user_row["num_trans"]}
- Monto total movido: {user_row["monto_total"]}
- Satisfacción: {user_row["satisfaccion_1_10"]}
- Cluster: {user_row["cluster"]}

ESTRATEGIA:
{estrategia}

ESTILO Y COMPORTAMIENTO:
- Adapta tu tono al estilo del usuario usando mensajes previos (formal, casual, corto, etc.)
- Si el usuario escribe corto, responde corto
- Si el usuario es casual, responde casual
- Evita sonar robótico o como reporte

PRIVACIDAD Y RELEVANCIA:
- NO menciones datos numéricos específicos del usuario a menos que sean necesarios para la respuesta
- NO enumeres todo el perfil sin razón
- Usa el contexto solo para mejorar recomendaciones, no para exponer información

RESPUESTAS:
- Sé útil, claro y accionable
- Prioriza consejos prácticos
- No inventes datos
"""

def deepseek_chat(messages, api_key, modelo="deepseek-chat", temp=0.7):
    url_api = "https://api.deepseek.com/v1/chat/completions"
    cuerpo_json = {"model": modelo, "messages": messages, "temperature": temp, "stream": False}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    respuesta = requests.post(url_api, headers=headers, data=json.dumps(cuerpo_json))
    if not respuesta.ok:
        raise RuntimeError(respuesta.text)
    return respuesta.json()["choices"][0]["message"]["content"]

# --- Modelos Pydantic ---
class ChatRequest(BaseModel):
    user_id: str
    mensaje: str

class ResetRequest(BaseModel):
    user_id: str

# --- Endpoints ---
@app.post("/chat")
def chat(req: ChatRequest):
    user_filtered = df[df["user_id"] == req.user_id]
    if user_filtered.empty:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    user_data = user_filtered.iloc[0]

    # Si es sesión nueva, inicializar con system prompt
    if req.user_id not in sesiones:
        contexto = construir_contexto(user_data)
        historial_previo = cargar_historial_previo(req.user_id)
        sesiones[req.user_id] = [{"role": "system", "content": contexto}] + historial_previo

    historial = sesiones[req.user_id]
    historial.append({"role": "user", "content": req.mensaje})

    respuesta = deepseek_chat(messages=historial, api_key=MI_CLAVE_DEEPSEEK)
    historial.append({"role": "assistant", "content": respuesta})

    return {"respuesta": respuesta, "user_id": req.user_id}

@app.post("/reset")
def reset(req: ResetRequest):
    sesiones.pop(req.user_id, None)
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}