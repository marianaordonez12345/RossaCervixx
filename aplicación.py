import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import datetime
import os

# --- 1. CONFIGURACIÓN Y ESTILO ROSA ---
st.set_page_config(page_title="RosaCervix AI", page_icon="🌸", layout="centered")

# Este bloque de código cambia el color de fondo y de los textos
st.markdown("""
    <style>
    .stApp {
        background-color: #FFF0F5; /* Rosa clarito (Lavender Blush) */
    }
    [data-testid="stSidebar"] {
        background-color: #FFE4E1; /* Rosa un poco más oscuro para la barra */
        border-right: 2px solid #FFC0CB;
    }
    h1, h2, h3, p {
        color: #D02090 !important; /* Color de letra rosa fuerte para resaltar */
    }
    .stButton>button {
        background-color: #FFB6C1;
        color: white;
        border-radius: 20px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF69B4;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌸 RosaCervix AI")
st.write("Detección Inteligente de Células para Prevención de Cáncer")

# --- 2. CARGA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    return YOLO("best.pt")

model = cargar_modelo()

# --- 3. FORMULARIO DE PACIENTE (Barra Lateral) ---
with st.sidebar:
    st.header("📋 Registro de Paciente")
    nombre = st.text_input("Nombre(s):")
    apellido = st.text_input("Apellidos:")
    edad = st.number_input("Edad:", min_value=0, max_value=120, step=1)
    st.divider()
    st.write("Completa los datos antes de subir la imagen.")

# --- 4. CARGA Y ANÁLISIS DE IMAGEN ---
archivo = st.file_uploader("Sube la imagen de la citología (JPG/PNG):", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo)
    st.image(img, caption="Imagen seleccionada", use_container_width=True)

    if st.button("🚀 INICIAR ANÁLISIS"):
        if not nombre or not apellido:
            st.warning("⚠️ Por favor, ingresa los datos del paciente a la izquierda.")
        else:
            with st.spinner("Analizando con Inteligencia Artificial..."):
                # Ejecutar detección
                results = model(img)
                # .plot() es lo que dibuja los cuadros (bounding boxes) automáticamente
                res_plotted = results[0].plot() 

                # Mostrar resultado con cuadros
                st.image(res_plotted, caption="Resultado del Análisis Visual", use_container_width=True)

                # Contar detecciones
                conteo = len(results[0].boxes)
                st.info(f"Se detectaron {conteo} células en la muestra.")

                # --- 5. GUARDAR EN HISTORIAL ---
                archivo_csv = "historial_pacientes.csv"
                nuevo_dato = {
                    "Fecha": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M")],
                    "Nombre": [nombre],
                    "Apellido": [apellido],
                    "Edad": [edad],
                    "Detecciones": [conteo]
                }
                df_nuevo = pd.DataFrame(nuevo_dato)
                
                # Guardar en el archivo (se añade al final si ya existe)
                df_nuevo.to_csv(archivo_csv, mode='a', header=not os.path.exists(archivo_csv), index=False)
                st.success(f"✅ Datos guardados para {nombre} {apellido}.")

# --- 6. VISUALIZAR HISTORIAL ---
if os.path.exists("historial_pacientes.csv"):
    if st.checkbox("Ver base de datos de pacientes"):
        df_hist = pd.read_csv("historial_pacientes.csv")
        st.dataframe(df_hist, use_container_width=True)