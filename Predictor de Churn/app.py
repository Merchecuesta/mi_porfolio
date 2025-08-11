
# ===============================================
# STREAMLIT APP CON FONDO, SIDEBAR CUSTOM E ÍNDICE
# ===============================================

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import joblib
from phik import phik_matrix
from phik.report import plot_correlation_matrix
from PIL import Image
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu
import base64
import streamlit as st
from streamlit_option_menu import option_menu


st.markdown(
    """
    <style>
    /* FUENTE GENERAL */
    html, body, [class*="css"] {
        font-family: 'Candara', 'Candara Light', Calibri, 'Segoe UI', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6, p, div, span {
        font-family: 'Candara', 'Candara Light', Calibri, 'Segoe UI', sans-serif !important;
    }

    /* FONDO EN TODA LA PÁGINA (incluyendo fuera del cuerpo) */
    body {
        background-image: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)),
                          url("https://raw.githubusercontent.com/Merchecuesta/ONLINE_DS_THEBRIDGE_MERCHECUESTA/main/fondo.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* OPCIONAL: fondo blanco semi-transparente para el contenido principal */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Menú + índice combinado
with st.sidebar:
    selected = option_menu(
        menu_title="Índice",
        options=[
            "Objetivo",
            "Datos",
            "Métricas",
            "Modelos",
            "Comparación entre Modelos",
            "Coste-Beneficio",
            "Predictor de Churn"
        ],
        icons=[
            "info-circle", "calculator", "bar-chart-line", "gear", "graph-up", "currency-dollar", "pie-chart"
        ],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#f0f2f6"},
            "icon": {"color": "#004d99", "font-size": "20px"},
            "nav-link": {
                "font-size": "18px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#99ccff",
                "color": "#004d99",
                "border-radius": "10px",
            },
            "nav-link-selected": {
                "background-color": "#004d99",
                "color": "white",
            },
        }
    )
df = pd.read_csv('ML/data/Telco_churn.csv')


# =========================
# FUNCIONES
# =========================

# Matriz de confusion
def mostrar_matriz_confusion(tn, fp, fn, tp, modelo):
    data = {
        "Predicción Negativa": [f"TN: {tn}", f"FN: {fn}"],
        "Predicción Positiva": [f"FP: {fp}", f"TP: {tp}"]
    }
    df_cm = pd.DataFrame(data, index=["Cliente NO dado de baja", "Cliente SÍ dado de baja"])
    
    st.subheader(f"Matriz de Confusión - {modelo}")
    st.table(df_cm)


# Metricas de negocio

def mostrar_metricas_negocio(precision0, recall0, f1_0, support0, 
                              precision1, recall1, f1_1, support1, 
                              accuracy, auc_roc):
    st.subheader("Métricas de negocio")
    
    df_metrics = pd.DataFrame({
        "Cliente NO dado de baja": [precision0, recall0, f1_0, support0],
        "Cliente SÍ dado de baja": [precision1, recall1, f1_1, support1]
    }, index=["Precisión", "Recall", "F1-score", "Nº de casos"])
    
    st.table(df_metrics)
    st.write(f"**Exactitud global (Accuracy):** {accuracy}")
    st.write(f"**AUC-ROC:** {auc_roc}")


# Mostrar contenido según menú
st.title(selected)

if selected == "Objetivo":
    st.subheader("Predicción de Abandono de Clientes en empresa de Telecomunicaciones")
    st.markdown("""
    El propósito principal de este proyecto es **anticipar el abandono de clientes (churn)** en una empresa del sector de telecomunicaciones.  
    A través del análisis de datos históricos sobre el comportamiento, perfil y contratación de servicios de los usuarios, buscamos:

    - **Identificar patrones y factores clave** asociados al abandono.
    - **Predecir con precisión** qué clientes tienen mayor probabilidad de dejar la compañía.
    - **Segmentar la base de clientes** según su riesgo de churn.
    - **Diseñar estrategias de retención personalizadas y efectivas**, enfocadas en reducir la pérdida de clientes y optimizar la rentabilidad.

    Este enfoque permitirá a la empresa **tomar decisiones proactivas** basadas en datos, mejorando tanto la fidelización como la eficiencia de las campañas de marketing.
    """)
    
elif selected == "Datos":
    st.markdown("""
    En la reunión inicial con el cliente, se analizó el conjunto de datos entregado, que contiene información detallada sobre **7,043** observaciones.  
    
    Los datos incluyen tanto variables demográficas como de los servicios contratados.
    
    Contamos con las siguientes columnas principales:

    | Columna           | Descripción                                                                                 |
    |-------------------|---------------------------------------------------------------------------------------------|
    | customerID      | Identificador único de cada cliente                                                        |
    | gender          | Género del cliente (Male / Female)                                                         |
    | SeniorCitizen   | Indica si el cliente es mayor de 65 años (1 = sí, 0 = no)                                      |
    | Partner         | Indica si el cliente tiene pareja (Yes / No)                                               |
    | Dependents      | Indica si el cliente tiene personas dependientes a su cargo (Yes / No)                     |
    | tenure          | Número de meses que el cliente lleva con la empresa                                       |
    | PhoneService    | Indica si el cliente tiene servicio telefónico (Yes / No)                                 |
    | MultipleLines   | Indica si el cliente tiene múltiples líneas telefónicas (Yes / No / No phone service)      |
    | InternetService | Tipo de servicio de internet contratado (DSL / Fiber optic / No)                          |
    | OnlineSecurity  | Servicio de seguridad online adicional (Yes / No / No internet service)                   |
    | OnlineBackup    | Servicio de respaldo online (Yes / No / No internet service)                              |
    | DeviceProtection| Plan de protección para dispositivos (Yes / No / No internet service)                     |
    | TechSupport     | Soporte técnico adicional (Yes / No / No internet service)                                |
    | StreamingTV     | Uso de servicio de streaming de TV (Yes / No / No internet service)                       |
    | StreamingMovies | Uso de servicio de streaming de películas (Yes / No / No internet service)                |
    | Contract        | Tipo de contrato (Month-to-month / One year / Two year)                                   |
    | PaperlessBilling| Uso de facturación sin papel (Yes / No)                                                   |
    | PaymentMethod   | Método de pago (Bank transfer / Credit card / Electronic check / Mailed check)            |
    | MonthlyCharges  | Cargo mensual que paga el cliente                                                         |
    | TotalCharges    | Total de cargos acumulados (puede tener valores faltantes si tenure es cero)                 |
    | Churn           | Variable objetivo: indica si el cliente abandonó la empresa (Yes / No)                    |
    
    Determinamos que el número de datos aportado es suficiente para entrenar a un modelo y preveemos que al tener el 'Churn' (lo que el cliente desea predecir) podemos inclinarnos por un tipo de modelo supervisado.
    """)

    st.subheader("Proporción de Clientes que Abandonan vs. Retienen")

    st.markdown("""
    Para comprender mejor el reto de predicción, es fundamental conocer cómo se distribuyen las clases según la variable objetivo (Churn).  
    Analizar esta proporción nos permite identificar si el modelo deberá enfrentarse a un **problema de clases balanceadas** o si, por el contrario, se trata de una **clase minoritaria**, lo que implicaría un escenario de **desbalance de clases**.  
    Esta información es clave para definir la estrategia de modelado y las métricas de evaluación más adecuadas.
    """)

    # Cálculo de conteos
    counts = df['Churn'].value_counts()
    labels = ['No', 'Sí']
    values = [counts['No'], counts['Yes']]
    colors = ['mediumseagreen', 'indianred']

    # Crear gráfico de pastel
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=16)
    )])

    fig.update_layout(
        title_text="Proporción de Clientes que Abandonan vs. Retienen",
        margin=dict(t=40, b=0, l=0, r=0)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Correlación y Multicolinealidad entre las variables")
    st.subheader("Heatmap de Correlación Phi_k")
    img = Image.open('phik_heatmap.png')
    st.image(img, use_container_width=True)

    st.markdown("""
    Como se puede apreciar en este mapa de calor, la mayoría de las variables son categóricas y muestran distintos grados de correlación entre ellas.  
    Esto significa que algunos atributos están relacionados o tienden a comportarse de manera similar, lo que puede influir en la construcción del modelo y en la interpretación de los resultados. 
    De manera contraria, vemos que hay algunos que apenas guardan relación con otros.  
    
    Entender estas relaciones es fundamental para seleccionar las variables más relevantes y evitar redundancias que puedan afectar la precisión del modelo.
    """)

elif selected == "Métricas":
    st.header("Métricas de Evaluación del Modelo")
    st.markdown("""
    Se propusieron las siguientes métricas para evaluar el desempeño del modelo y su impacto en negocio:

    | Métrica                 | ¿Qué mide?                                                         | ¿Por qué es útil para su negocio?                           |
    |-------------------------|------------------------------------------------------------------|------------------------------------------------------------|
    | **Recall (Sensibilidad)** | De todos los clientes que se van, ¿cuántos detectamos?           | Detecta la fuga real. Prioridad alta para retención         |
    | **Precision**           | De los clientes predichos como churn, ¿cuántos realmente se van? | Evita costes innecesarios y molestias a clientes            |
    | **F1 Score**            | Balance entre precision y recall                                 | Equilibrio para detectar churn sin alarmar en exceso        |
    | **Matriz de confusión** | Verdaderos/falsos positivos y negativos                          | Interpretación clara de errores y aciertos                   |
    | **Accuracy**            | Proporción de predicciones correctas en total                    | Medida general de efectividad. Poco útil en clases desbalanceadas                                |
    | **ROC AUC Score**       | Capacidad del modelo para distinguir las clases                  | Métrica global, aunque menos intuitiva para negocio          |
    """)

    st.markdown("""
    El cliente da máxima importancia al **Recall** para asegurarse de detectar la mayor cantidad posible de clientes que podrían abandonar la compañía.  
    
    Sin embargo, también se buscará un buen equilibrio entre precisión y recall para seleccionar el modelo más adecuado, evitando falsas alarmas innecesarias y optimizando la efectividad de las acciones de retención.
    
    Finalmente se decide que el modelo con mejor **ROC** y buen **Recall** será el elegido
    """)



elif selected == "Modelos":
    st.header("Análisis de Modelos de Clasificación")

    # ---------------------------
    # MODELO 1: Regresión Logística
    # ---------------------------
    with st.expander("Regresión Logística"):
        st.header("""
        **Interpretación del modelo:**

        La Regresión Logística logra detectar bastantes clientes que se dan de baja (recall de 0.78 en la clase positiva),
        aunque comete muchos falsos positivos (clientes que no se dan de baja pero el modelo dice que sí).

        La precisión para los que se dan de baja es baja (0.47), lo que indica que muchos de los que el modelo predice como baja no lo son realmente.

        El AUC-ROC de 0.816 sugiere que el modelo tiene buena capacidad discriminativa general.
        """)

        mostrar_matriz_confusion(tn=1057, fp=495, fn=123, tp=438, modelo="Regresión Logística")

        mostrar_metricas_negocio(
            precision0=0.90, recall0=0.68, f1_0=0.77, support0=1552,
            precision1=0.47, recall1=0.78, f1_1=0.59, support1=561,
            accuracy=0.71, auc_roc=0.816
        )

    # ---------------------------
    # MODELO 2: Random Forest
    # ---------------------------
    with st.expander("Random Forest"):
        st.header("""
        **Interpretación del modelo:**

        Random Forest mejora el equilibrio entre precisión y recall. Detecta muchos clientes que se dan de baja (recall de 0.76) 
        y también reduce la cantidad de falsos positivos respecto al modelo anterior.

        Su precisión en la clase positiva mejora hasta 0.55 y tiene un mejor `f1-score`.

        AUC-ROC sube a 0.842, indicando una mejor discriminación general.
        """)

        mostrar_matriz_confusion(tn=1201, fp=351, fn=135, tp=426, modelo="Random Forest")

        mostrar_metricas_negocio(
            precision0=0.90, recall0=0.77, f1_0=0.83, support0=1552,
            precision1=0.55, recall1=0.76, f1_1=0.64, support1=561,
            accuracy=0.77, auc_roc=0.842
        )

    # ---------------------------
    # MODELO 3: CatBoost
    # ---------------------------
    with st.expander("CatBoost"):
        st.header("""
        **Interpretación del modelo:**

        CatBoost mantiene un recall alto en clientes que se dan de baja (0.79), similar a los anteriores, pero consigue mejor equilibrio.

        Aunque su precisión en la clase positiva es algo baja (0.52), destaca por su rendimiento estable.

        El AUC-ROC es el mejor de los tres (0.852), lo que lo convierte en un modelo muy competitivo para detectar bajas.
        """)

        mostrar_matriz_confusion(tn=757, fp=278, fn=78, tp=296, modelo="CatBoost")

        mostrar_metricas_negocio(
            precision0=0.91, recall0=0.73, f1_0=0.81, support0=1035,
            precision1=0.52, recall1=0.79, f1_1=0.62, support1=374,
            accuracy=0.75, auc_roc=0.852
        )

    # ---------------------------
    # MODELO 4: LightGBM
    # ---------------------------
    with st.expander("LightGBM"):
        st.header("""
        **Interpretación del modelo:**

        LightGBM ofrece un excelente equilibrio entre precisión y recall, logrando detectar correctamente a la mayoría de clientes que se dan de baja (recall de 0.82) y manteniendo una precisión aceptable.

        Tiene un `AUC-ROC` de **0.834**, indicando una muy buena capacidad de clasificación.
        """)

        mostrar_matriz_confusion(tn=717, fp=316, fn=69, tp=305, modelo="LightGBM")

        mostrar_metricas_negocio(
            precision0=0.9122, recall0=0.6941, f1_0=0.7883, support0=1033,
            precision1=0.4911, recall1=0.8155, f1_1=0.6131, support1=374,
            accuracy=0.7264, auc_roc=0.8338
        )

    # ---------------------------
    # MODELO 5: XGBoost
    # ---------------------------
    with st.expander("XGBoost"):
        st.header("""
        **Interpretación del modelo:**

        XGBoost destaca por su robustez. Con recall de **0.82** en la clase de baja y una precisión razonable de **0.49**, logra un `AUC-ROC` de **0.841**, el segundo más alto de todos los modelos.

        Es una opción competitiva para predecir bajas con buena precisión global.
        """)

        mostrar_matriz_confusion(tn=1075, fp=477, fn=101, tp=460, modelo="XGBoost")

        mostrar_metricas_negocio(
            precision0=0.91, recall0=0.69, f1_0=0.79, support0=1552,
            precision1=0.49, recall1=0.82, f1_1=0.61, support1=561,
            accuracy=0.73, auc_roc=0.841
        )

    # ---------------------------
    # MODELO 6: K-Means (No Supervisado)
    # ---------------------------
    with st.expander("K-Means (No Supervisado)"):
        st.header("""
        **Interpretación del modelo:**

        Aunque K-Means no usa la variable objetivo (Churn) para entrenar, los clusters formados muestran diferencias interesantes.

        - El **Cluster 0** tiene un `churn` del **31.9%**, mientras que el **Cluster 1** solo del **7.4%**.
        - El modelo segmenta a los clientes en perfiles: modernos/digitales vs. tradicionales.

        **Matriz de confusión entre cluster y churn real:**
        """)

        mostrar_matriz_confusion(tn=3756, fp=1407, fn=1756, tp=113, modelo="K-Means")

        st.markdown("""
        **Conclusiones clave:**

        - **Cluster 0** → Clientes modernos, pagan más, usan streaming y facturación electrónica → **alto riesgo de baja**.
        - **Cluster 1** → Clientes tradicionales, contratos largos, sin servicios extra → **bajo riesgo de baja**.

         **Estrategia**:
        - Retener a los clientes del Cluster 0 (valiosos pero con riesgo).
        - Mantener satisfechos a los del Cluster 1 (estables, aunque menos rentables).
        """)


elif selected == "Comparación entre Modelos":
    st.subheader("Comparativa de Métricas entre Modelos")

    import pandas as pd

    metricas_df = pd.DataFrame({
        "Modelo": [
            "Regresión Logística",
            "Random Forest",
            "CatBoost",
            "LightGBM",
            "XGBoost",
            "K-Means (No Supervisado)"
        ],
        "Accuracy": [0.71, 0.77, 0.75, 0.7264, 0.73, 0.55],
        "Precision": [0.47, 0.55, 0.52, 0.4911, 0.49, 0.07],
        "Recall": [0.78, 0.76, 0.79, 0.8155, 0.82, 0.06],
        "F1-Score": [0.59, 0.64, 0.62, 0.6131, 0.61, 0.07],
        "AUC-ROC": [0.816, 0.842, 0.852, 0.8338, 0.841, None]
    })

    st.dataframe(metricas_df.style.format(precision=3), use_container_width=True)

    st.markdown("""
    ### Conclusiones Finales

    - **CatBoost** se posiciona como el modelo **más equilibrado y robusto**: logra el **mayor AUC-ROC (0.852)**, lo que indica la **mejor capacidad para discriminar entre clientes que abandonan y los que no**, y mantiene un **recall muy competitivo (0.79)**. Esto lo convierte en una opción especialmente confiable para identificar clientes en riesgo sin comprometer la calidad general del modelo.

    - **XGBoost** destaca por tener el **mayor recall (0.82)**, es decir, detecta ligeramente más clientes que se darán de baja, aunque con un **AUC-ROC ligeramente inferior (0.841)**, lo que sugiere un menor equilibrio global.

    - **LightGBM** también muestra un rendimiento destacable, con un **recall de 0.8155** y **AUC-ROC de 0.834**, lo que lo convierte en una alternativa válida si se prioriza velocidad y eficiencia computacional.

    - **K-Means**, al ser un algoritmo no supervisado, **no es útil directamente para predecir el churn**, pero **aporta valor en la segmentación** de clientes por perfiles de riesgo, lo que puede enriquecer estrategias de retención.

    - Modelos como **Random Forest** y **Regresión Logística** quedan rezagados en términos de recall y AUC-ROC, aunque podrían ser útiles en entornos donde la interpretabilidad del modelo sea clave.

    ---

    ### **Recomendación Final para decisión de Negocio**

    Si el objetivo principal es **maximizar la detección de clientes en riesgo (recall)** manteniendo un **alto nivel de discriminación general** y **consistencia en las métricas clave**:
    
    **CatBoost es la mejor elección**  
    Ofrece **la mayor capacidad predictiva global (AUC-ROC = 0.852)**, con un **recall competitivo (0.79)** y una **excelente estabilidad en métricas**. Es especialmente recomendable si se busca un modelo sólido y balanceado para producción.
    
    **XGBoost**, si bien obtiene un recall ligeramente mayor (0.82), presenta una **pérdida de equilibrio global**, por lo que es recomendable **solo si el negocio prioriza exclusivamente la sensibilidad** por encima de todo lo demás.
    
    **LightGBM** es una alternativa eficiente, muy útil en entornos donde el **tiempo de entrenamiento o escalabilidad** sea un factor clave, aunque ligeramente por debajo de CatBoost en términos de rendimiento.

    ---

    ### **Respuesta de Negocio**
    
    La elección recomendada es: **CatBoost**.

    **Ventajas clave**:

    -  **Menos falsos positivos**: se reducen los errores al identificar clientes fieles como si fueran a darse de baja.
    -  **Campañas de retención más eficientes**: se enfocan en quienes realmente están en riesgo, optimizando recursos.
    -  **Mejor experiencia de cliente**: se evitan acciones innecesarias sobre clientes satisfechos, lo que mejora la relación con la marca.
    """)

elif selected == "Coste-Beneficio":
    st.header("Simulación Coste/Beneficio campañas Marketing ajustadas al modelo")

    st.markdown("""
    Basándonos en la información proporcionada por el departamento de Marketing, llevamos a cabo una simulación económica con el objetivo de evaluar el coste/beneficio de las campañas de marketing, usando nuestro modelo:
    """)

    # Tabla resumen de parámetros
    parametros = pd.DataFrame({
        "Parámetro": [
            "Ingreso medio mensual", "Coste de contacto por cliente", "Tiempo adicional retenido",
            "Porcentaje éxito retención"
        ],
        "Valor": ["70 EUR", "10 EUR", "4 meses", "30 %"],
        "Descripción": [
            "Mediana de MonthlyCharges", "Llamada, mensaje u oferta", "Estimación media",
            "Clientes que deciden quedarse tras contacto"
        ]
    })
    st.table(parametros)

    opcion_campana = st.radio("Campañas:", ["Retención Si/No", "Segmentación", "Comparativa entre ellas"])

    if opcion_campana == "Retención Si/No":
        st.subheader("Campaña de retención centrada solo en si el cliente es churn o no")
        st.subheader("**Resultados sobre test set (1.409 clientes):**")
        res_test = pd.DataFrame({
            "Concepto": [
                "Clientes reales con churn", "Recall del modelo", "Precisión del modelo",
                "Clientes contactados por campaña", "Verdaderos positivos (churn real)", "Falsos positivos (no churn)"
            ],
            "Valor": [374, "79%", "52%", 574, 296, 278]
        })
        st.table(res_test)

        # Cálculo económico
        st.markdown("**Cálculo económico:**")
        economico = pd.DataFrame({
            "Concepto": [
                "Coste total campaña", "Clientes retenidos", "Ingresos salvados",
                "Beneficio neto", "ROI"
            ],
            "Valor": [
                "5.740 EUR", "83", "23.240 EUR", "17.500 EUR", "3.05"
            ],
            "Cálculo": [
                "574 clientes x 10 EUR", "296 x 30 %", "83 clientes x 280 EUR", "23.240 - 5.740 EUR", "17.500 / 5.740"
            ]
        })
        st.table(economico)

        st.subheader("Visualización Coste vs Ingresos Salvados Campaña Si/No")

        df_grafico = pd.DataFrame({
            "Concepto": ["Coste de la Campaña", "Ingresos Salvados"],
            "EUR": [5740, 23240]
        })
        palette = ["#e74c3c", "#27ae60"]

        fig1, ax1 = plt.subplots(figsize=(4, 3))
        sns.barplot(data=df_grafico, x="Concepto", y="EUR", palette=palette, ax=ax1)

        ax1.set_ylabel("EUR", fontsize=8)
        ax1.set_title("Coste de campaña vs Ingresos salvados", fontsize=9)
        ax1.tick_params(axis='x', labelsize=7)
        ax1.tick_params(axis='y', labelsize=7)

        # Anotaciones dentro de las barras
        for p in ax1.patches:
            altura = p.get_height()
            ax1.text(
                p.get_x() + p.get_width() / 2,
                altura / 2,
                f'{altura:,.2f} EUR',
                ha='center',
                va='center',
                color='black',
                fontsize=7,
                fontweight='bold'
            )

        st.pyplot(fig1)

    elif opcion_campana == "Segmentación":
        st.subheader("Campaña de retención basada en segmentación por probabilidad de churn")
        st.markdown("""
        Proposición segmentación clientes según la probabilidad de churn para adaptar el gasto en campañas según riesgo.
        """)

        # Tabla de rangos
        rango_prob = pd.DataFrame({
            "Rango probabilidad": ["0.85 - 1.00", "0.65 - 0.84", "0.40 - 0.64"],
            "Etiqueta de riesgo": ["Alto riesgo", "Riesgo moderado-alto", "Riesgo medio"],
            "Interpretación": [
                "Cliente muy probable de abandonar",
                "Requiere atención inmediata",
                "Posible abandono si no se actúa"
            ]
        })
        st.markdown("**Rangos y etiquetas:**")
        st.table(rango_prob)

        # Tabla de costes por riesgo
        st.markdown("**Costes por segmento:**")
        costes_segmento = pd.DataFrame({
            "Nivel riesgo": ["Alto riesgo", "Moderado-alto", "Riesgo medio", "Total"],
            "Clientes": [123, 288, 286, 697],
            "Coste medio (EUR)": [10.47, 5.46, 1.82, ""],
            "Coste total (EUR)": [1287.81, 1572.48, 520.52, 3380.81]
        })
        st.table(costes_segmento)

        # Tabla ingresos salvados
        st.markdown("**Ingresos salvados y beneficios:**")
        beneficios_segmento = pd.DataFrame({
            "Nivel riesgo": ["Alto riesgo", "Moderado-alto", "Riesgo medio", "Total"],
            "Clientes retenidos": [37, 86, 86, 209],
            "Ingresos salvados (EUR)": [10360, 24080, 24080, 58520]
        })
        st.table(beneficios_segmento)

        # Tabla resumen ROI
        st.markdown("**Resumen:**")
        resumen_seg = pd.DataFrame({
            "Concepto": ["Coste total campaña", "Ingresos salvados", "Beneficio neto", "ROI"],
            "Valor (EUR)": ["3.380,81", "58.520", "55.139,19", "16,3"]
        })
        st.table(resumen_seg)

        # 🎯 Gráfico comparación ingresos por riesgo
        st.header("Visualización Ingresos vs Coste de Campaña Segmentación")

        coste_total = 1287.81 + 1572.48 + 520.52  # = 3380.81 EUR
        ingresos_totales = 10360 + 24080 + 24080  # = 58520 EUR

        labels = ["Coste Total Campaña", "Ingresos Salvados"]
        valores = [coste_total, ingresos_totales]
        colores = ['#e74c3c', '#27ae60']  # rojo para coste, verde para ingresos

        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(labels, valores, color=colores)

        ax.set_ylabel("EUR", fontsize=8)
        ax.set_title("Coste Campaña vs Ingresos Salvados Segmentación", fontsize=9)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        # Anotaciones dentro de las barras
        for bar in bars:
            altura = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                altura / 2,
                f'{altura:,.2f} EUR',
                ha='center',
                va='center',
                color='black',
                fontsize=7,
                fontweight='bold'
            )

        st.pyplot(fig)

    elif opcion_campana == "Comparativa entre ellas":
        st.subheader("Comparación visual de las dos campañas de publicidad")

        # Datos
        labels = ['Coste Campaña', 'Ingresos Salvados']
        campaña_segmentacion = [1287.81 + 1572.48 + 520.52, 10360 + 24080 + 24080]  # [3380.81, 58520]
        campaña_siono = [5740, 23240]

        x = np.arange(len(labels))  # posiciones 0,1
        width = 0.35  # ancho barras

        fig, ax = plt.subplots(figsize=(6, 4))

        # Barras lado a lado
        bars1 = ax.bar(x - width / 2, campaña_segmentacion, width, label='Campaña Segmentación', color='#2980b9')
        bars2 = ax.bar(x + width / 2, campaña_siono, width, label='Campaña Sí/No', color='#27ae60')

        # Etiquetas y título
        ax.set_ylabel('EUR', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title('Comparación de Costes e Ingresos entre Campañas', fontsize=12)
        ax.legend(fontsize=9)
        ax.tick_params(axis='y', labelsize=8)

        # Función para poner anotaciones dentro de las barras
        def annotate_bars(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height / 2,
                    f'{height:,.0f} EUR',
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8,
                    fontweight='bold'
                )

        annotate_bars(bars1)
        annotate_bars(bars2)

        # Líneas que conectan las barras de cada categoría
        for i in range(len(x)):
            # Obtener las coordenadas centrales superiores de cada barra
            x1 = bars1[i].get_x() + bars1[i].get_width() / 2
            y1 = bars1[i].get_height()
            x2 = bars2[i].get_x() + bars2[i].get_width() / 2
            y2 = bars2[i].get_height()
            # Dibujar línea entre barras
            ax.plot([x1, x2], [y1, y2], 'k--', lw=1)

        st.pyplot(fig)

        st.subheader("Conclusión de la Comparación de Campañas")
        st.markdown("""
        La comparación visual entre la campaña de segmentación y la campaña tradicional “Sí/No” revela diferencias significativas en términos de coste y beneficio económico.

        - **Coste de la campaña:** La campaña de segmentación tiene un coste total considerablemente menor (~3.380 EUR) frente a la campaña Sí/No (~5.740 EUR), lo que indica un uso más eficiente del presupuesto.
          
        - **Ingresos salvados:** Por otro lado, la campaña de segmentación genera ingresos salvados más altos (~58.520 EUR) en comparación con la campaña Sí/No (~23.240 EUR), casi el doble, demostrando una mayor efectividad en retener clientes.

        - **Relación coste-beneficio:** Esto se traduce en un retorno de inversión mucho más favorable para la campaña de segmentación, lo que sugiere que invertir en estrategias diferenciadas y segmentadas permite optimizar recursos y maximizar resultados.

        En resumen, la campaña de segmentación es claramente la opción preferible para maximizar beneficios, reducir gastos y mejorar la eficiencia de las acciones de retención de clientes.
        """)

 

elif selected == "Predictor de Churn":
    st.title("Predicción de Churn con tu CSV")
    st.markdown("Carga un archivo CSV con datos de tus clientes y obtén la predicción de abandono.")

    # Clase para limpieza numérica
    class NumericCleaner(BaseEstimator, TransformerMixin):
        def __init__(self, columns):
            self.columns = columns
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            X = X.copy()
            for col in self.columns:
                X[col] = pd.to_numeric(X[col].replace(' ', np.nan), errors='coerce').astype('float64')
            return X

    uploaded_file = st.file_uploader("Carga tu archivo CSV", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            customer_ids = df["customerID"].copy()

            # Columnas a eliminar (coincidir con pipeline)
            cols_drop = [
                'customerID', 'DeviceProtection', 'StreamingTV', 'gender',
                'PhoneService', 'Dependents', 'TechSupport', 'StreamingMovies'
            ]
            df = df.drop(columns=[col for col in cols_drop if col in df.columns])

            # Cargar pipeline pre-entrenado (asegúrate que pipeline_churn.joblib está en el mismo directorio)
            pipeline = joblib.load("pipeline_churn.joblib")

            # Predecir probabilidades y clases
            probs = pipeline.predict_proba(df)[:, 1]
            preds = pipeline.predict(df)

            # Mostrar resultados en dataframe
            resultado = pd.DataFrame({
                "customerID": customer_ids,
                "Probabilidad_Churn": probs,
                "Prediccion_Churn": preds
            })

            st.success("✅ Predicción completada.")
            st.dataframe(resultado)

            # Botón para descargar resultados
            csv = resultado.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar resultados como CSV",
                data=csv,
                file_name='predicciones_churn.csv',
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"⚠️ Error al procesar el archivo: {e}")
