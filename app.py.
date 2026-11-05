import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="Clasificador Cervicouterino - RosaCervix", layout="centered")

st.title("🔬 Detector y Clasificador Citológico - RosaCervix")
st.write("Sube una imagen de citología para realizar la detección y clasificación basada en el Sistema Bethesda.")

@st.resource_cache if hasattr(st, "resource_cache") else st.cache_resource
def cargar_modelo():
    # Carga tu modelo entrenado que está en la raíz del repositorio
    return YOLO("best.pt")

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# Componente con clave única para evitar conflictos en la web
uploaded_file = st.file_uploader("Elige una imagen de citología...", type=["jpg", "jpeg", "png"], key="subida_imagen_cervix")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada correctamente.", use_column_width=True)
    
    if st.button("Ejecutar Análisis y Detección", key="btn_analizar"):
        with st.spinner("Procesando imagen con IA..."):
            # Convertir imagen para YOLO
            img_array = np.array(image)
            results = model(img_array)
            
            # Mostrar resultados visuales
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Resultado de la Segmentación y Clasificación", use_column_width=True)
            st.success("¡Análisis completado con éxito!")
        
