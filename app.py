import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.set_page_config(page_title="Predicción de Rendimiento Escolar", layout="centered")

st.title("Predicción de Aprobación de Estudiantes")
st.write("Introduce los hábitos y notas del estudiante para calcular la probabilidad de aprobación con Deep Learning.")

# Cargar recursos previamente entrenados
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('modelo_mlp_estudiantes.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("¡Modelo y Escalador cargados correctamente!")
except Exception as e:
    st.error(f"Error al cargar artefactos: {e}")

# Entradas del usuario
st.header("Características Académicas y Hábitos")

# Generamos campos según las variables del dataset
val1 = st.slider("Nota de Tareas (0-100)", 0.0, 100.0, 70.0)
val2 = st.slider("Nota de Exámenes (0-100)", 0.0, 100.0, 70.0)
val3 = st.slider("Porcentaje de Asistencia (0-100)", 0.0, 100.0, 85.0)
val4 = st.number_input("Horas de Estudio Semanales", min_value=0.0, max_value=50.0, value=15.0)
val5 = st.number_input("Promedio Horas de Sueño Diarias", min_value=3.0, max_value=12.0, value=7.0)
val6 = st.selectbox("Actividades Extracurriculares", options=[0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
val7 = st.slider("Calificación Final de Referencia", 0.0, 100.0, 70.0)

if st.button("Predecir Estado del Estudiante"):
    input_data = np.array([[val1, val2, val3, val4, val5, val6, val7]])
    input_scaled = scaler.transform(input_data)

    # Predecir con MLP
    probabilidad = model.predict(input_scaled)[0][0]
    resultado = "Aprobado" if probabilidad >= 0.5 else "Reprobado"
    color = "green" if resultado == "Aprobado" else "red"

    st.markdown(f"### Resultado: <span style='color:{color}'>{resultado}</span>", unsafe_allow_html=True)
    st.metric(label="Probabilidad de Aprobación", value=f"{probabilidad * 100:.2f}%")
