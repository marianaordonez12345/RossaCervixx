import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import datetime
import os
from fpdf import FPDF

# --- CONFIGURACIÓN Y ESTILO ---
st.set_page_config(page_title="RosaCervix AI", page_icon="🌸", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #FFF0F5; }
    h1, h2, h3, label, .stMarkdown, .stCheckbox { 
        color: #8B008B !important; 
        font-weight: bold !important;
    }
    [data-testid="stSidebar"] { background-color: #FFC0CB; }
    .stButton>button { 
        background-color: #D87093 !important; 
        color: white !important; 
        border-radius: 20px; 
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIÓN PARA GENERAR PDF PROFESIONAL ---
def generar_pdf(nombre, apellido, edad, conteo, notas):
    pdf = FPDF()
    pdf.add_page()
    
    # Logo Rosa
    pdf.set_text_color(255, 105, 180) 
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(200, 15, txt="RosaCervix AI", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    
    # Encabezado
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="DIRECCION DE PRESTACIONES MEDICAS", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 8, txt="INFORME DE RESULTADOS DE CITOLOGIA DIGITAL", ln=True, align='C')
    pdf.ln(10)
    
    # Datos del Paciente
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(255, 240, 245)
    pdf.cell(200, 10, txt="DATOS DEL PACIENTE", ln=True, fill=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, txt=f"PACIENTE: {nombre.upper()} {apellido.upper()}", ln=True)
    pdf.cell(200, 8, txt=f"EDAD: {edad} años", ln=True)
    pdf.cell(200, 8, txt=f"FECHA: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.ln(10)
    
    # Lógica de Diagnóstico con Semáforo
    if conteo == 0:
        diagnostico = "NEGATIVO: No se detectaron estructuras atípicas."
        riesgo = "BAJO"
    elif conteo < 5:
        diagnostico = "SOSPECHOSO LEVE: Se detectaron pocas estructuras."
        riesgo = "MODERADO"
    else:
        diagnostico = "ALERTA: Se detectaron múltiples estructuras. REQUIERE REVISIÓN."
        riesgo = "ALTO"

    # Sección de Resultados
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="DIAGNOSTICO E INTERPRETACION", ln=True, fill=True)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 8, txt=f"Nivel de riesgo sugerido: {riesgo}", ln=True)
    pdf.multi_cell(0, 7, txt=f"Resultado: {diagnostico}")
    
    # Notas del Analista
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="OBSERVACIONES DEL ANALISTA", ln=True, fill=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, txt=notas if notas else "Sin observaciones adicionales.")

    # Nota Legal
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, txt="IMPORTANTE: Este sistema es una herramienta de apoyo al triaje. El diagnóstico definitivo debe ser emitido exclusivamente por un médico patólogo tras la revisión de la muestra original.")
    
    archivo_pdf = f"reporte_{nombre}_{apellido}.pdf"
    pdf.output(archivo_pdf)
    return archivo_pdf

# --- INTERFAZ ---
st.title("🌸 RosaCervix AI")
model = YOLO("best.pt")

with st.sidebar:
    st.header("📋 Registro de Paciente")
    nombre = st.text_input("Nombre(s):")
    apellido = st.text_input("Apellidos:")
    edad = st.number_input("Edad:", min_value=0, max_value=120)
    notas = st.text_area("Observaciones adicionales:")

archivo = st.file_uploader("Sube la imagen de la muestra:", type=["jpg", "png", "jpeg"])

if archivo:
    img = Image.open(archivo)
    st.image(img, caption="Imagen seleccionada", use_container_width=True)

    if st.button("🚀 ANALIZAR Y GENERAR PDF"):
        if not nombre or not apellido:
            st.warning("⚠️ Por favor, ingresa los datos del paciente.")
        else:
            with st.spinner("Procesando con IA..."):
                results = model(img)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Resultado del Análisis", use_container_width=True)
                conteo = len(results[0].boxes)
                
                # Guardar en CSV
                archivo_csv = "historial_pacientes.csv"
                nuevo_dato = {"Fecha": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M")], "Nombre": [nombre], "Apellido": [apellido], "Edad": [edad], "Detecciones": [conteo]}
                pd.DataFrame(nuevo_dato).to_csv(archivo_csv, mode='a', header=not os.path.exists(archivo_csv), index=False)
                
                # Generar PDF
                archivo_pdf = generar_pdf(nombre, apellido, edad, conteo, notas)
                with open(archivo_pdf, "rb") as f:
                    st.download_button("📥 Descargar Reporte PDF", f, file_name=archivo_pdf)
                st.success("✅ Análisis completado.")

if os.path.exists("historial_pacientes.csv"):
    if st.checkbox("Ver base de datos de pacientes"):
        st.dataframe(pd.read_csv("historial_pacientes.csv"), use_container_width=True)
        
