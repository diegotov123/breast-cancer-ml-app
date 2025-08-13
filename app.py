import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico de Cáncer de Mama - ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .prediction-benign {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
    }
    .prediction-malignant {
        background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar y entrenar el modelo
@st.cache_data
def load_and_train_model():
    # Cargar dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calcular precisión
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, X.columns, accuracy, X_test, y_test

# Cargar modelo y datos
model, scaler, feature_names, accuracy, X_test, y_test = load_and_train_model()

# Header principal
st.markdown('<h1 class="main-header">🩺 Diagnóstico de Cáncer de Mama con ML</h1>', unsafe_allow_html=True)

# Advertencia médica
st.markdown("""
<div class="warning-box">
    <h3>⚠️ IMPORTANTE - Solo para fines educativos</h3>
    <p><strong>Esta aplicación es únicamente para demostración educativa y NO debe usarse para diagnósticos médicos reales.</strong></p>
    <p>Siempre consulte con un profesional médico calificado para cualquier inquietud de salud.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar con información del modelo
st.sidebar.header("📊 Información del Modelo")
st.sidebar.markdown(f"""
<div class="metric-card">
    <h3>Precisión del Modelo</h3>
    <h2>{accuracy:.2%}</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
**Características del modelo:**
- Algoritmo: Random Forest
- Dataset: Wisconsin Breast Cancer
- Características: 30 variables
- Clases: Benigno / Maligno
""")

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["🔬 Diagnóstico", "📈 Análisis", "📋 Información"])

with tab1:
    st.header("Realizar Diagnóstico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Opción Rápida - Valores Predefinidos")
        
        # Botones para casos de ejemplo
        if st.button("📝 Cargar Ejemplo Benigno", type="secondary"):
            st.session_state.example_type = "benign"
            
        if st.button("📝 Cargar Ejemplo Maligno", type="secondary"):
            st.session_state.example_type = "malignant"
            
        # Botón para valores aleatorios
        if st.button("🎲 Generar Valores Aleatorios"):
            st.session_state.example_type = "random"
    
    with col2:
        st.subheader("⚡ Predicción Rápida")
        
        # Obtener ejemplo basado en la selección
        if 'example_type' in st.session_state:
            if st.session_state.example_type == "benign":
                # Ejemplo típico de tumor benigno
                example_idx = np.where(y_test == 1)[0][0]  # Benigno = 1
            elif st.session_state.example_type == "malignant":
                # Ejemplo típico de tumor maligno
                example_idx = np.where(y_test == 0)[0][0]  # Maligno = 0
            else:
                # Valor aleatorio
                example_idx = np.random.randint(0, len(X_test))
                
            example_data = X_test.iloc[example_idx].values
            
            # Realizar predicción
            example_scaled = scaler.transform([example_data])
            prediction = model.predict(example_scaled)[0]
            probability = model.predict_proba(example_scaled)[0]
            
            # Mostrar resultado
            if prediction == 1:  # Benigno
                st.markdown(f"""
                <div class="prediction-benign">
                    ✅ DIAGNÓSTICO: BENIGNO<br>
                    Probabilidad: {probability[1]:.1%}
                </div>
                """, unsafe_allow_html=True)
            else:  # Maligno
                st.markdown(f"""
                <div class="prediction-malignant">
                    ⚠️ DIAGNÓSTICO: MALIGNO<br>
                    Probabilidad: {probability[0]:.1%}
                </div>
                """, unsafe_allow_html=True)
            
            # Mostrar gráfico de probabilidades
            fig = go.Figure(data=[
                go.Bar(x=['Maligno', 'Benigno'], 
                      y=[probability[0], probability[1]],
                      marker_color=['#e74c3c', '#27ae60'])
            ])
            fig.update_layout(
                title="Probabilidades de Diagnóstico",
                yaxis_title="Probabilidad",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Entrada manual de datos
    st.subheader("🔧 Configuración Manual (Avanzado)")
    
    with st.expander("Introducir valores manualmente"):
        st.write("Introduce los valores de las características principales:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            radius_mean = st.number_input("Radio promedio", min_value=0.0, max_value=50.0, value=14.0)
            texture_mean = st.number_input("Textura promedio", min_value=0.0, max_value=50.0, value=19.0)
            perimeter_mean = st.number_input("Perímetro promedio", min_value=0.0, max_value=200.0, value=92.0)
            area_mean = st.number_input("Área promedio", min_value=0.0, max_value=2500.0, value=655.0)
            
        with col2:
            smoothness_mean = st.number_input("Suavidad promedio", min_value=0.0, max_value=1.0, value=0.096)
            compactness_mean = st.number_input("Compacidad promedio", min_value=0.0, max_value=1.0, value=0.104)
            concavity_mean = st.number_input("Concavidad promedio", min_value=0.0, max_value=1.0, value=0.089)
            concave_points_mean = st.number_input("Puntos cóncavos promedio", min_value=0.0, max_value=1.0, value=0.048)
            
        with col3:
            symmetry_mean = st.number_input("Simetría promedio", min_value=0.0, max_value=1.0, value=0.181)
            fractal_dimension_mean = st.number_input("Dimensión fractal promedio", min_value=0.0, max_value=1.0, value=0.063)
        
        # Para simplicidad, usar valores promedio para el resto de características
        if st.button("🔍 Realizar Diagnóstico Manual"):
            # Crear array con las 10 características principales y completar con valores promedio
            manual_features = [
                radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean
            ]
            
            # Completar con valores promedio para las otras 20 características
            avg_features = X_test.mean().values[10:]
            full_features = manual_features + list(avg_features)
            
            # Realizar predicción
            manual_scaled = scaler.transform([full_features])
            manual_prediction = model.predict(manual_scaled)[0]
            manual_probability = model.predict_proba(manual_scaled)[0]
            
            if manual_prediction == 1:
                st.success(f"✅ Diagnóstico: BENIGNO (Probabilidad: {manual_probability[1]:.1%})")
            else:
                st.error(f"⚠️ Diagnóstico: MALIGNO (Probabilidad: {manual_probability[0]:.1%})")

with tab2:
    st.header("📈 Análisis del Dataset")
    
    # Cargar datos completos para análisis
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['diagnosis'] = data.target
    df['diagnosis_name'] = df['diagnosis'].map({0: 'Maligno', 1: 'Benigno'})
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de diagnósticos
        diagnosis_counts = df['diagnosis_name'].value_counts()
        fig1 = px.pie(values=diagnosis_counts.values, names=diagnosis_counts.index, 
                     title="Distribución de Diagnósticos")
        fig1.update_traces(marker_colors=['#e74c3c', '#27ae60'])
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Comparación de características principales
        feature_to_plot = st.selectbox(
            "Selecciona una característica para analizar:",
            ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
        )
        
        fig2 = px.box(df, x='diagnosis_name', y=feature_to_plot, 
                     title=f"Distribución de {feature_to_plot} por Diagnóstico")
        fig2.update_traces(marker_color='#3498db')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Correlación entre características
    st.subheader("Matriz de Correlación (Características Principales)")
    main_features = [col for col in df.columns if 'mean' in col][:10]
    corr_matrix = df[main_features].corr()
    
    fig3 = px.imshow(corr_matrix, 
                    title="Correlación entre Características",
                    color_continuous_scale="RdYlBu")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.header("📋 Información Detallada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 Sobre el Dataset")
        st.markdown("""
        **Wisconsin Breast Cancer Dataset:**
        - **Fuente**: UCI Machine Learning Repository
        - **Muestras**: 569 casos
        - **Características**: 30 variables numéricas
        - **Clases**: 2 (Benigno/Maligno)
        
        **Características medidas:**
        - Radio (distancia media del centro a puntos del perímetro)
        - Textura (desviación estándar de valores de escala de grises)
        - Perímetro
        - Área
        - Suavidad (variación local en longitudes de radio)
        - Compacidad (perímetro² / área - 1.0)
        - Concavidad (severidad de porciones cóncavas del contorno)
        - Puntos cóncavos (número de porciones cóncavas del contorno)
        - Simetría
        - Dimensión fractal ("aproximación de costa" - 1)
        """)
    
    with col2:
        st.subheader("🤖 Sobre el Modelo")
        st.markdown(f"""
        **Random Forest Classifier:**
        - **Algoritmo**: Ensemble de árboles de decisión
        - **Precisión**: {accuracy:.2%}
        - **Ventajas**:
          - Resistente al sobreajuste
          - Maneja bien características numéricas
          - Proporciona importancia de características
        
        **Métricas de Rendimiento:**
        """)
        
        # Mostrar métricas detalladas
        y_pred = model.predict(scaler.transform(X_test))
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        metrics_df = pd.DataFrame({
            'Métrica': ['Precisión', 'Recall', 'F1-Score', 'Accuracy'],
            'Valor': [f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", f"{accuracy:.3f}"]
        })
        
        st.dataframe(metrics_df, hide_index=True)
    
    st.markdown("---")
    st.subheader("⚠️ Limitaciones y Consideraciones")
    st.markdown("""
    **Importante recordar:**
    - Este modelo es solo para demostración educativa
    - Los datos reales de diagnóstico requieren múltiples pruebas y análisis profesional
    - La medicina personalizada considera muchos más factores
    - Siempre consulte con profesionales médicos calificados
    - Los resultados pueden variar según la calidad de los datos de entrada
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🩺 Aplicación de Demostración - Diagnóstico de Cáncer de Mama con ML</p>
    <p><em>Solo para fines educativos - No usar para diagnósticos reales</em></p>
</div>
""", unsafe_allow_html=True)
