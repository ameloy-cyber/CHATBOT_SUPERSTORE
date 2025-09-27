# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 09:47:40 2025

@author: Sala_603
"""

import pandas as pd
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

EXCEL_FILE = r"C:\Users\Sala_\Downloads\superstore.xlsx"
if os.path.exists(EXCEL_FILE):
    df = pd.read_excel(EXCEL_FILE)
    print("Base cargada correctamente ✅")
    print(df.head())
else:
    print("⚠️ No se encontró el archivo superstore.xlsx")

def save_interaction(user_msg, bot_response):
    """Guarda la conversación en Excel"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame({
        "timestamp": [timestamp],
        "usuario": [user_msg],
        "bot": [bot_response]
    })
    if os.path.exists(EXCEL_FILE):
        df_existing = pd.read_excel(EXCEL_FILE)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new
    df_final.to_excel(EXCEL_FILE, index=False)

# ------------------------------
# Base FAQ
# ------------------------------
faq_data = {
    "ventas": "📊 Las ventas totales alcanzaron **$228.8M** en Texas y **$225.8M** en California como los estados líderes.",
    "ganancias": "💰 Las ganancias más altas se registraron en **New York con $473.6M**, mientras que Texas tuvo una pérdida de **-$160.9M**.",
    "top_sales": "🏆 El estado con mayores ventas fue **Texas con $228.8M**, seguido de **California con $225.8M**.",
    "resumen_ventas": "📝 En general, algunos estados muestran fuertes ingresos en ventas, pero varios como Texas y Pennsylvania reportan pérdidas en ganancias.",

    "entregas": "📦 El tiempo promedio de entrega es de **9.8 días para Office Supplies**, **8.9 días para Furniture** y **8.5 días para Technology**.",
    "mejoras": "🚀 Se recomienda optimizar los tiempos de entrega en Office Supplies, ya que son los más largos.",
    "comparacion": "📊 Office Supplies tiene el mayor tiempo de entrega (**+1 día** sobre Furniture y Technology).",
    "resumen_entregas": "📝 En general, los tiempos son similares entre categorías, pero hay oportunidad de mejora clara en Office Supplies.",

    "productos_top": "🏆 Los estados con mejores resultados son California y New York, donde los productos más vendidos también son rentables.",
    "productos_problema": "⚠️ En Texas, las ventas son altas pero con pérdidas millonarias. Recomendación: ajustar precios o analizar productos poco rentables.",
    "estrategia_descuentos": "💡 Aplicar descuentos mayores en productos de baja rotación y limitarlos en productos de alta demanda para mejorar la rentabilidad.",

    "saludo": "🤖 ¡Hola! Soy tu asistente de Superstore. Pregúntame sobre ventas, ganancias, tiempos de entrega o recomendaciones de mejora, productos, descuentos."
}

training_phrases = {
    "ventas": ["dime las ventas", "cómo van las ventas", "ventas por estado", "informe de ventas", "total de ventas acumuladas", "ventas más altas", "ventas de california", "ventas de texas"],
    "ganancias": ["cómo van las ganancias", "ganancias por estado", "informe de ganancias", "estados con pérdidas", "ganancias acumuladas", "cuáles fueron las mayores ganancias"],
    "top_sales": ["quién tuvo más ventas", "top ventas", "estado con más ventas", "ranking de ventas", "ventas más altas en estados"],
    "resumen_ventas": ["dame un resumen de ventas", "resumen general de ganancias", "balance de ventas y ganancias", "cómo estuvo el rendimiento", "resumen de ventas"],

    "entregas": ["tiempos de entrega", "promedio de entregas", "cuánto tardan los envíos", "tiempo de entrega por categoría", "duración promedio de entregas"],
    "mejoras": ["dónde podemos mejorar", "recomendaciones de mejora", "qué categoría tarda más", "oportunidades de mejora", "mejorar tiempos de entrega"],
    "comparacion": ["comparar tiempos de entrega", "qué categoría es más rápida", "cuál tarda más en entregarse", "diferencias de entrega entre categorías"],
    "resumen_entregas": ["resumen de entregas", "balance de tiempos", "estado actual de los envíos", "reporte de entregas", "resumen general de entregas"],

    "productos_top": ["productos más rentables", "productos estrella", "qué productos funcionan mejor", "top productos"],
    "productos_problema": ["productos con pérdidas", "productos poco rentables", "qué productos revisar", "problemas con productos"],
    "estrategia_descuentos": ["descuentos", "qué estrategia de descuentos seguir", "descuentos recomendados", "cómo aplicar promociones"],

    "saludo": ["hola", "buenas", "qué tal", "hey", "saludos"],
}

# ------------------------------
# Entrenamiento NLP
# ------------------------------
X, y = [], []
for intent, phrases in training_phrases.items():
    for phrase in phrases:
        X.append(phrase)
        y.append(intent)

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
model = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X_vec)

def predict_intent(user_input):
    user_vec = vectorizer.transform([user_input])
    dist, idx = model.kneighbors(user_vec)
    intent = y[idx[0][0]]
    confidence = 1 - dist[0][0]
    return intent if confidence >= 0.5 else None

# ------------------------------
# Simulación en consola
# ------------------------------
print("\n🤖 ChatBot Superstore (modo consola). Escribe 'salir' para terminar.\n")

while True:
    user_input = input("👤 Tú: ")
    if user_input.lower() == "salir":
        break
    intent = predict_intent(user_input)
    if intent:
        response = faq_data[intent]
    else:
        response = "❓ No entendí tu consulta."
    print("🤖 Bot:", response)
    save_interaction(user_input, response)
