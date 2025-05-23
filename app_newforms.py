import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials
import json
import pandas as pd
import joblib
import hashlib
import uuid
from fpdf import FPDF
from io import BytesIO  
from streamlit_javascript import st_javascript
from streamlit_geolocation import streamlit_geolocation
from streamlit_folium import folium_static
from streamlit.components.v1 import html
import re
from datetime import datetime
import time
import gspread
import numpy as np
from gspread.exceptions import APIError

# --- CONFIGURACIONES GLOBALES ---
st.set_page_config(page_title="DIABETO", page_icon="🏥", layout="wide")
RUTA_PREGUNTAS = "preguntas2.json"
COLUMNAS_MODELO = ['sexo', 'edad', 'a0201', 'a0206', 'a0601', 'a0602a',
    'a0602b', 'a0602c', 'a0602d', 'a0701a', 'a0701b', 'a0703', 'a0704', 
    'a0801a', 'a0803a', 'a0804a', 'a0806a', 'a0801b', 'a0803b', 'a0804b', 
    'a0806b', 'a0801c', 'a0803c', 'a0804c', 'a0806c', 'a1401', 'a1405',
    'peso', 'talla', 'cintura']

# Importación de modelo
RUTA_MODELO1 = "1_modelo.pkl"
RUTA_MODELO2 = "2_modelo.pkl"

# API PLACE MAPA
@st.cache_resource
def cargar_modelo1():
    return joblib.load(RUTA_MODELO1)

@st.cache_resource
def cargar_modelo2():
    return joblib.load(RUTA_MODELO2)

# --- FUNCIONES AUXILIARES ---
def leer_en_voz(texto: str):
    if st.session_state.get("voz_activa", False):
        html(f"""
            <script>
                window.speechSynthesis.cancel();  // Cancela cualquier lectura en curso
                var mensaje = new SpeechSynthesisUtterance("{texto}");
                mensaje.lang = "es-MX";
                mensaje.pitch = 1;
                mensaje.rate = 0.95;
                window.speechSynthesis.speak(mensaje);
            </script>
        """, height=0)

def set_background():
    st.markdown("""
        <style>
            .stApp {
                background-color: white;
            }
        </style>
    """, unsafe_allow_html=True)

def cargar_css(path):
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def conectar_google_sheet(nombre=None, key=None, debug=False):
    try:
        # Validar credenciales
        if "gcp_service_account" not in st.secrets:
            raise RuntimeError("No se encontró 'gcp_service_account' en st.secrets")

        creds_dict = st.secrets["gcp_service_account"]
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        # Crear credenciales
        creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)
        client = gspread.authorize(creds)

        # Abrir hoja por clave o nombre
        if key:
            sheet = client.open_by_key(key).sheet1
        elif nombre:
            sheet = client.open(nombre).sheet1
        else:
            raise ValueError("Debes proporcionar 'key' o 'nombre' para abrir el Google Sheet")

        if debug:
            st.success(f"✅ Conectado exitosamente a: {sheet.title}")

        return sheet

    except gspread.exceptions.SpreadsheetNotFound:
        raise RuntimeError("🔒 No se pudo encontrar el Google Sheet. ¿La clave es correcta y está compartido con la cuenta de servicio?")
    except gspread.exceptions.APIError as api_error:
        raise RuntimeError(f"🌐 Error de API de Google Sheets: {str(api_error)}")
    except Exception as e:
        raise RuntimeError(f"❌ Error inesperado al conectar con Google Sheets: {str(e)}")

def render_pregunta(pregunta, key):
    tipo = pregunta.get("tipo", "text")
    label = pregunta.get("label", "").strip()

    if not label:
        label = "Pregunta sin título"  # Fallback obligatorio
        st.warning(f"⚠️ Pregunta con 'label' vacío detectada en key: {key}")

    if tipo == "text":
        return st.text_input(label, key=key)
    elif tipo == "number":
        return st.number_input(label, key=key)
    elif tipo == "textarea":
        return st.text_area(label, key=key)
    elif tipo == "select":
        opciones = ["Selecciona"] + pregunta.get("opciones", [])
        seleccion = st.selectbox(label, opciones, key=key)
        if "valores" in pregunta and seleccion != "Selecciona":
            return pregunta["valores"][pregunta["opciones"].index(seleccion)]
        return "" if seleccion == "Selecciona" else seleccion
    else:
        return st.text_input(label, key=key)  # Fallback para tipos no reconocidos


def obtener_variables_importantes(modelo, datos, top_n=5):
    modelo_final = None
    if hasattr(modelo, "named_steps"):
        for step in modelo.named_steps.values():
            if hasattr(step, "feature_importances_"):
                modelo_final = step
                break
    elif hasattr(modelo, "feature_importances_"):
        modelo_final = modelo

    if modelo_final is None:
        st.warning("⚠️ El modelo no tiene 'feature_importances_'.")
        return []

    importancias = modelo_final.feature_importances_
    top_indices = importancias.argsort()[::-1]
    fila = datos.iloc[0].to_dict()

    variables_relevantes = []
    for i in top_indices[:top_n]:
        codigo = COLUMNAS_MODELO[i]
        valor = fila.get(codigo, "")
        variables_relevantes.append((codigo, valor))

    return variables_relevantes

def generar_pdf(respuestas_completas, variables_relevantes):
    """
    Genera un PDF con las respuestas del paciente y las variables más relevantes.

    Args:
        respuestas_completas (list): Lista de tuplas (pregunta, respuesta)
        variables_relevantes (list): Lista de tuplas (pregunta, valor)

    Returns:
        BytesIO: Buffer del PDF generado listo para descarga.
    """
    pdf = FPDF()
    pdf.set_title("Evaluación de Riesgo - DIABETO")
    pdf.set_author("Sistema DIABETO")

    # Página 1 - Respuestas completas
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "📝 Respuestas del Paciente", ln=True)

    pdf.set_font("Arial", "", 10)
    if respuestas_completas:
        for pregunta, respuesta in respuestas_completas:
            try:
                pregunta = str(pregunta).encode('latin-1', 'ignore').decode('latin-1')
                respuesta = str(respuesta).encode('latin-1', 'ignore').decode('latin-1')
                pdf.multi_cell(0, 8, f"{pregunta}: {respuesta}")
            except Exception as e:
                pdf.multi_cell(0, 8, f"[Error al mostrar respuesta: {e}]")
    else:
        pdf.cell(0, 10, "No se registraron respuestas.", ln=True)

    # Página 2 - Variables más relevantes
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "🔍 Variables Más Relevantes", ln=True)

    pdf.set_font("Arial", "", 10)
    if variables_relevantes:
        for pregunta, respuesta in variables_relevantes:
            try:
                pregunta = str(pregunta).encode('latin-1', 'ignore').decode('latin-1')
                respuesta = str(respuesta).encode('latin-1', 'ignore').decode('latin-1')
                pdf.multi_cell(0, 8, f"{pregunta}: {respuesta}")
            except Exception as e:
                pdf.multi_cell(0, 8, f"[Error al mostrar variable: {e}]")
    else:
        pdf.cell(0, 10, "No se identificaron variables destacadas.", ln=True)

    # Convertir a bytes
    buffer = BytesIO()
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    buffer.write(pdf_bytes)
    buffer.seek(0)
    return buffer

def obtener_hoja_usuarios(debug=False):
    """
    Devuelve la hoja de cálculo de usuarios conectada por clave.

    Args:
        debug (bool): Si es True, muestra detalles del proceso en Streamlit.

    Returns:
        gspread.Worksheet: Objeto de hoja de cálculo, o None si falla.
    """
    try:
        key = st.secrets["google_sheets"]["usuarios_key"]
        hoja = conectar_google_sheet(key=key, debug=debug)

        if debug:
            st.success(f"✅ Hoja de usuarios conectada: {hoja.title}")

        return hoja

    except KeyError:
        st.error("❌ No se encontró 'usuarios_key' en st.secrets['google_sheets']")
        return None
    except Exception as e:
        st.error(f"❌ Error al conectar con la hoja de usuarios: {str(e)}")
        return None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def buscar_usuario_por_nombre(nombre):
    sheet = obtener_hoja_usuarios()
    registros = sheet.get_all_records()
    for row in registros:
        if row["Nombre Completo"].strip().lower() == nombre.strip().lower():
            return row
    return None

def registrar_usuario(nombre, password):
    sheet = obtener_hoja_usuarios()
    sheet.append_row([nombre.strip(), hash_password(password)])

# --- FUNCIONES PRINCIPALES ---
def login_page():
    set_background()
    cargar_css("style.css")

    st.markdown("""
        <div class='form-container'>
            <div style='text-align:center; margin-bottom:25px;'>
                <h1 style='color:black;'>DIABETO<br>
                Queremos ayudarte a saber si tienes señales que podrían indicar riesgo de diabetes tipo 2. 
                Es rápido y fácil.</h1>
            </div>
    """, unsafe_allow_html=True)

    # Activar modo voz si no está definido
    if "voz_activa" not in st.session_state:
        st.session_state["voz_activa"] = False

    st.session_state["voz_activa"] = st.checkbox(
        "🗣️ ¿Deseas activar el modo de lectura en voz alta?",
        value=st.session_state["voz_activa"]
    )

    if st.session_state["voz_activa"]:
        leer_en_voz("Bienvenido a DIABETO. Queremos ayudarte a saber si tienes señales que podrían indicar riesgo de diabetes tipo dos. Es rápido y fácil.")
        leer_en_voz("Selecciona una opción. Puedes iniciar sesión si ya tienes cuenta, o crear una cuenta nueva.")

    modo = st.radio("Selecciona una opción:", ["Selecciona una opción", "Iniciar sesión", "Crear cuenta"])

    if modo == "Iniciar sesión":
        if st.session_state["voz_activa"]:
            leer_en_voz("Por favor, escribe tu nombre completo y tu contraseña para iniciar sesión.")

        with st.form("login_form"):
            nombre = st.text_input("Nombre completo", key="login_nombre")
            password = st.text_input("Contraseña", type="password", key="login_pass")

            if st.form_submit_button("Ingresar"):
                try:
                    usuario = buscar_usuario_por_nombre(nombre)
                    if usuario and usuario.get("Contraseña Hasheada") == hash_password(password):
                        st.session_state["logged_in"] = True
                        st.session_state["usuario"] = nombre
                        st.sidebar.markdown(f"👤 Sesión activa: **{nombre}**")
                        st.success(f"Bienvenido, {nombre}")
                        if st.session_state["voz_activa"]:
                            leer_en_voz(f"Bienvenido, {nombre}. Has iniciado sesión correctamente.")
                        st.rerun()
                    else:
                        st.error("No pudimos encontrar tus datos. Revisa que estén bien escritos o intenta registrarte.")
                        if st.session_state["voz_activa"]:
                            leer_en_voz("No pudimos encontrar tus datos. Intenta de nuevo o crea una cuenta.")
                except Exception as e:
                    st.error(f"❌ Error durante el inicio de sesión: {str(e)}")

    elif modo == "Crear cuenta":
        if st.session_state["voz_activa"]:
            leer_en_voz("Por favor, escribe tu nombre completo y una contraseña para crear una cuenta nueva.")

        with st.form("registro_form"):
            nombre = st.text_input("Nombre completo", key="reg_nombre")
            password = st.text_input("Contraseña", type="password", key="reg_pass")

            if st.form_submit_button("Registrar"):
                try:
                    if buscar_usuario_por_nombre(nombre):
                        st.error("Este nombre ya fue usado. Prueba con uno diferente.")
                        if st.session_state["voz_activa"]:
                            leer_en_voz("Ese nombre ya fue usado. Prueba con uno diferente.")
                    elif not nombre or not password:
                        st.warning("Te falta llenar algún dato. Revisa por favor.")
                        if st.session_state["voz_activa"]:
                            leer_en_voz("Te falta llenar algún dato. Revisa por favor.")
                    else:
                        registrar_usuario(nombre, password)
                        st.success("Cuenta creada correctamente. Ya puedes iniciar sesión.")
                        if st.session_state["voz_activa"]:
                            leer_en_voz("Cuenta creada correctamente. Ya puedes iniciar sesión.")
                        
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error al registrar usuario: {str(e)}")
                    if st.session_state["voz_activa"]:
                        leer_en_voz("Ocurrió un error al registrar tu cuenta.")

    st.markdown("</div>", unsafe_allow_html=True)

def mostrar_perfil():
    st.title("👩🏽👨🏽 Mi Cuenta")

    # Estilo visual
    st.markdown("""
        <style>
            .perfil-container {
                text-align: center;
                margin-top: 35px;
            }
            .texto-introductorio {
                font-size: 22px;
                line-height: 1.6;
                padding: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='perfil-container'>", unsafe_allow_html=True)

    # Obtener nombre del usuario
    nombre_usuario = st.session_state.get("usuario", "Usuario")
    st.markdown(f"<h2>{nombre_usuario}</h2>", unsafe_allow_html=True)

    # Leer nombre en voz si está activado
    if st.session_state.get("voz_activa", False):
        leer_en_voz(f"Bienvenido, {nombre_usuario}. Esta es tu cuenta.")

    # Cargar texto de introducción
    texto_crudo = ""
    try:
        with open("intro_text.json", encoding="utf-8") as f:
            textos = json.load(f)
            texto_crudo = textos.get("mi_cuenta", "").strip()
            if not texto_crudo:
                texto_crudo = "Aquí podrás ver tu perfil y tus evaluaciones."

    except FileNotFoundError:
        texto_crudo = "No se encontró el archivo de texto introductorio."
    except json.JSONDecodeError:
        texto_crudo = "Error al leer el archivo de introducción. Verifica que el formato JSON sea válido."

    # Convertir Markdown simple a HTML
    texto_html = texto_crudo \
        .replace("**", "<b>").replace("*", "<i>") \
        .replace("</i><b>", "</i></b><b>") \
        .replace("</b><i>", "</b></i><i>")

    st.markdown(f"<div class='texto-introductorio'>{texto_html}</div>", unsafe_allow_html=True)

    # Leer en voz si corresponde
    if st.session_state.get("voz_activa", False) and texto_crudo:
        texto_sin_html = re.sub(r'<[^>]+>', '', texto_crudo)
        leer_en_voz(texto_sin_html.strip())

    st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data
def predecir_nuevos_registros(df_input, threshold1=0.18, threshold2=0.18):
    """
    Versión robusta con manejo de valores vacíos y errores
    """
    try:
        # Validación inicial del dataframe
        if df_input.empty or not isinstance(df_input, pd.DataFrame):
            st.error("Datos de entrada inválidos o vacíos")
            return None

        # Crear copia segura
        df = df_input.copy()
        
        # 1. Limpieza y conversión de datos
        # ---------------------------------
        # Mapeo de sexo seguro
        if 'sexo' in df.columns:
            df['sexo'] = df['sexo'].apply(
                lambda x: 1 if str(x).strip().lower() in ['hombre', '1', 'h', 'masculino'] else
                         2 if str(x).strip().lower() in ['mujer', '2', 'm', 'femenino'] else
                         -1  # Valor por defecto para inválidos
            )

        # Columnas numéricas - manejo seguro de vacíos
        numeric_cols = ['edad', 'peso', 'talla', 'cintura'] + [c for c in COLUMNAS_MODELO if c.startswith('a')]
        for col in numeric_cols:
            if col in df.columns:
                # Convertir a numérico, vacíos se convierten a NaN y luego a -1
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1)

        # Asegurar todas las columnas requeridas
        for col in COLUMNAS_MODELO:
            if col not in df.columns:
                df[col] = -1  # Valor por defecto

        # 2. Predicción con Modelo 1
        # --------------------------
        modelo1 = cargar_modelo1()
        try:
            X1 = df[COLUMNAS_MODELO].astype(float)
            df["Probabilidad Estimada 1"] = modelo1.predict_proba(X1)[:, 1]
            df["Predicción Óptima 1"] = (df["Probabilidad Estimada 1"] >= threshold1).astype(int)
        except Exception as e:
            st.error(f"Fallo Modelo 1: {str(e)}")
            return None

        # 3. Predicción con Modelo 2 (solo si Modelo 1 detecta riesgo)
        # -----------------------------------------------------------
        if df["Predicción Óptima 1"].iloc[0] == 1:
            modelo2 = cargar_modelo2()
            try:
                X2 = df[COLUMNAS_MODELO].astype(float)
                proba2 = modelo2.predict_proba(X2)[:, 1]
                
                # Validar probabilidades
                if np.isnan(proba2).any():
                    raise ValueError("Probabilidad contiene NaN")
                
                df["Probabilidad Estimada 2"] = proba2
                df["Predicción Óptima 2"] = (df["Probabilidad Estimada 2"] >= threshold2).astype(int)
            except Exception as e:
                st.warning(f"Fallo Modelo 2: {str(e)} - Se mantienen resultados del Modelo 1")
                df["Probabilidad Estimada 2"] = None
                df["Predicción Óptima 2"] = None

        return df

    except Exception as e:
        st.error(f"Error crítico en predicción: {str(e)}")
        return None
    
def guardar_respuesta_paciente(fila_dict):
    """
    Guarda un registro de paciente en Google Sheets con control de errores,
    manejo de encabezados y verificación posterior.

    Args:
        fila_dict (dict): Diccionario con los datos del paciente.

    Returns:
        bool: True si se guardó correctamente, False si hubo error.
    """

    MAX_INTENTOS = 3
    intentos = 0

    while intentos < MAX_INTENTOS:
        try:
            if not fila_dict or not isinstance(fila_dict, dict):
                st.error("❌ Los datos del paciente están vacíos o mal formateados.")
                return False

            # Normalizar todos los valores a string
            fila_limpia = {
                str(k).strip(): str(v).strip() if v is not None else ""
                for k, v in fila_dict.items()
            }

            # Conectar a la hoja
            sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
            encabezados = sheet.row_values(1)

            # Crear encabezados si no existen
            if not encabezados:
                encabezados = list(fila_limpia.keys())
                sheet.update('A1', [encabezados])

            # Detectar y añadir columnas nuevas
            nuevas_columnas = [col for col in fila_limpia if col not in encabezados]
            if nuevas_columnas:
                encabezados += nuevas_columnas
                sheet.update('A1', [encabezados])

            # Ordenar datos según encabezados
            fila_ordenada = [fila_limpia.get(col, "") for col in encabezados]

            # Guardar la fila
            sheet.append_row(fila_ordenada)

            # Verificación simple por última fila exacta
            ultima_fila = sheet.row_values(sheet.row_count)
            if fila_ordenada != ultima_fila:
                raise ValueError("⚠️ La fila guardada no coincide con los datos enviados.")

            st.toast("✅ Datos guardados correctamente.")
            return True

        except APIError as e:
            intentos += 1
            st.warning(f"🌐 Error de conexión con Google Sheets (intento {intentos}/{MAX_INTENTOS}): {e}")
            time.sleep(2)

        except Exception as e:
            st.error(f"❌ Error al guardar: {str(e)}")
            return False

    return False
    
def mostrar_resultado_prediccion(pred: int, modelo_usado: int, variables_importantes: list = None) -> str:

    try:
        # Validación básica
        pred = int(round(pred))  # Asegurar que sea 0 o 1
        if modelo_usado not in [1, 2]:
            raise ValueError("modelo_usado debe ser 1 o 2.")

        # Determinar resultado
        if modelo_usado == 2:
            diagnostico = "Prediabético" if pred == 0 else "Diabético"
            color = "#FFA500" if pred == 0 else "#FF0000"
            emoji = "🟠" if pred == 0 else "🚨"
            mensaje = (
                "Tus respuestas indican señales compatibles con una condición prediabética. "
                "Te recomendamos consultar a un especialista."
                if pred == 0 else
                "Tus respuestas indican señales compatibles con diabetes tipo 2. "
                "Es importante que acudas a un centro de salud lo antes posible."
            )
        else:
            diagnostico = "Sano" if pred == 0 else "En Riesgo"
            color = "#4CAF50" if pred == 0 else "#FFA500"
            emoji = "✅" if pred == 0 else "⚠️"
            mensaje = (
                "¡Buenas noticias! No encontramos señales claras de diabetes. "
                "Sigue cuidando tu salud." if pred == 0 else
                "Tus respuestas muestran factores de riesgo. "
                "Te sugerimos una evaluación más detallada."
            )

        # Mostrar en interfaz
        st.markdown(f"""
            <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; 
                        border-left: 5px solid {color}; margin-bottom:20px;'>
                <h3 style='color:{color}; margin-top:0;'>{emoji} Diagnóstico: {diagnostico}</h3>
                <p style='margin-bottom:0;'>{mensaje}</p>
            </div>
        """, unsafe_allow_html=True)

        # Factores más relevantes
        texto_a_leer = mensaje
        if pred == 1 and variables_importantes:
            st.markdown("#### 🔍 Factores más relevantes en esta evaluación:")
            texto_a_leer += " Los factores más relevantes fueron: "
            for var, val in variables_importantes:
                st.markdown(f"- **{var}**: {val}")
                texto_a_leer += f"{var}, "

        # Leer en voz si está activado
        if st.session_state.get("voz_activa", False):
            leer_en_voz(texto_a_leer.strip())

        return diagnostico

    except Exception as e:
        st.error(f"❌ Error al mostrar el diagnóstico: {str(e)}")
        return "Diagnóstico no disponible"

def ejecutar_prediccion():
    """
    Aplica el modelo 2 a la última fila registrada en Google Sheets y muestra el diagnóstico.
    """
    try:
        # Obtener hoja y datos
        sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
        registros = sheet.get_all_records()
        if not registros:
            st.warning("No hay registros disponibles para analizar.")
            return

        df = pd.DataFrame(registros).dropna(how="all")
        if df.empty:
            st.warning("No hay registros válidos en la hoja.")
            return

        # Verificar columnas necesarias
        faltantes = [col for col in COLUMNAS_MODELO if col not in df.columns]
        if faltantes:
            st.error(f"Faltan columnas necesarias para el modelo: {faltantes}")
            return

        # Seleccionar última fila y convertir tipos
        X = df.iloc[[-1]][COLUMNAS_MODELO].replace("", -1)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(-1)

        # Cargar modelo y predecir
        modelo = cargar_modelo2()
        proba = modelo.predict_proba(X)[0, 1]
        pred = int(proba >= 0.21)

        # Variables importantes
        variables = obtener_variables_importantes(modelo, X)

        # Mostrar resultado
        mostrar_resultado_prediccion(pred, modelo_usado=2, variables_importantes=variables)

    except Exception as e:
        st.error(f"❌ Error durante la ejecución de la predicción: {str(e)}")
        if st.session_state.get("voz_activa", False):
            leer_en_voz("Ocurrió un error al procesar la predicción.")

def verificar_campos_faltantes(fila_dict):
    columnas_requeridas = [
        "Registrado por", "sexo", "edad", "a0201", "a0206", "a0601",
        "a0602a", "a0602b", "a0602c", "a0602d", "a0701a", "a0701b", "a0703", "a0704",
        "a0801a", "a0803a", "a0804a", "a0806a", "a0801b", "a0803b", "a0804b", "a0806b",
        "a0801c", "a0803c", "a0804c", "a0806c", "a1401", "a1405",
        "peso", "talla", "cintura",
        "Probabilidad Estimada 1", "Predicción Óptima 1",
        "Probabilidad Estimada 2", "Predicción Óptima 2"
    ]
    faltantes = [col for col in columnas_requeridas if col not in fila_dict or str(fila_dict[col]).strip() == ""]
    if faltantes:
        st.error(f"❌ Faltan columnas requeridas antes de guardar: {faltantes}")
        return False
    return True

def nuevo_registro():
    st.title("📝 Registro de Pacientes")

    if st.session_state.get("voz_activa", False):
        leer_en_voz("Estás en la sección de registro de pacientes. Por favor responde las siguientes preguntas.")

    # 1. Cargar estructura del formulario
    try:
        with open(RUTA_PREGUNTAS, encoding="utf-8") as f:
            secciones = json.load(f)
    except FileNotFoundError:
        st.error("No se encontró el archivo de preguntas.")
        return
    except json.JSONDecodeError:
        st.error("El archivo de preguntas está mal formado.")
        return

    respuestas = {}
    key_form = f"form_registro_{st.session_state.get('usuario', 'anon')}_{int(time.time())}"

    # 2. Mostrar formulario
    with st.form(key=key_form, clear_on_submit=True):
        for titulo, contenido in secciones.items():
            st.subheader(titulo)
            if st.session_state.get("voz_activa", False):
                leer_en_voz(titulo)

            if isinstance(contenido, dict):  # Antecedentes familiares
                for familiar, grupo in contenido.items():
                    with st.expander(f"Antecedentes familiares: {familiar}"):
                        for p in contenido:
                            if not isinstance(p, dict):
                                st.warning(f"⚠️ Entrada inválida en sección {titulo}: {p}")
                                continue

                            if not p.get("label"):
                                st.warning(f"⚠️ Pregunta sin 'label' detectada en código: {p.get('codigo', 'sin_codigo')}")
                                p["label"] = "Pregunta sin título"

                            if not p.get("codigo"):
                                st.warning(f"⚠️ Pregunta sin 'codigo', se omitirá.")
                                continue

                            codigo = p["codigo"]
                            respuestas[codigo] = render_pregunta(p, key=f"{key_form}_{codigo}")
            elif isinstance(contenido, list):  # Sección normal
                for p in contenido:
                    codigo = p.get("codigo", f"{uuid.uuid4().hex[:6]}")
                    respuestas[codigo] = render_pregunta(p, key=f"{key_form}_{codigo}")

        submitted = st.form_submit_button("💾 Guardar y Evaluar")

    # 3. Procesar envío
    if submitted:
        with st.spinner("Guardando y evaluando..."):
            try:
                # Validar campos obligatorios
                campos_requeridos = ['edad', 'sexo', 'peso', 'talla']
                faltantes = [c for c in campos_requeridos if respuestas.get(c) in [None, "", " "]]
                if faltantes:
                    raise ValueError(f"Faltan campos obligatorios: {', '.join(faltantes)}")

                # Convertir respuestas a DataFrame
                df_input = pd.DataFrame([respuestas])
                resultado = predecir_nuevos_registros(df_input)

                if resultado is None or resultado.empty:
                    raise RuntimeError("No se pudo completar la evaluación.")

                # Determinar modelo y predicción usados
                if "Predicción Óptima 2" in resultado.columns and not pd.isna(resultado["Predicción Óptima 2"].iloc[0]):
                    modelo_usado = 2
                    pred = int(resultado["Predicción Óptima 2"].iloc[0])
                else:
                    modelo_usado = 1
                    pred = int(resultado["Predicción Óptima 1"].iloc[0])

                modelo = cargar_modelo2() if modelo_usado == 2 else cargar_modelo1()
                variables_relevantes = obtener_variables_importantes(modelo, resultado)

                # Asegura columnas del modelo 2 aunque no se usen
                for col in ["Probabilidad Estimada 2", "Predicción Óptima 2"]:
                    if col not in fila_final:
                        fila_final[col] = ""

                # Preparar datos para guardar
                fila_final = resultado.iloc[0].to_dict()
                fila_final.update({
                    "Registrado por": st.session_state.get("usuario", "Anónimo"),
                    "Fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                # Guardar en Google Sheets
                if not verificar_campos_faltantes(fila_final):
                    st.stop()   
                exito = guardar_respuesta_paciente(fila_final)

                if exito:
                    st.success("✅ Respuestas guardadas y evaluadas.")
                    mostrar_resultado_prediccion(pred, modelo_usado, variables_importantes=variables_relevantes)
                else:
                    st.error("❌ No se pudieron guardar los datos.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                if st.session_state.get("voz_activa", False):
                    leer_en_voz("Ocurrió un error al guardar los datos.")

def mostrar_pacientes():
    st.title("📋 Participante")

    if st.session_state.get("voz_activa", False):
        leer_en_voz("Estás en la sección de participantes. Aquí puedes consultar los registros guardados.")

    try:
        sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
        registros = sheet.get_all_records()
        if not registros:
            st.info("No hay registros disponibles.")
            return

        df = pd.DataFrame(registros).dropna(how="all")
        usuario_actual = st.session_state.get("usuario", "").strip().lower()
        df = df[df["Registrado por"].str.strip().str.lower() == usuario_actual]

        if df.empty:
            st.info("No tienes registros guardados aún.")
            return

        df["ID"] = [f"Registro #{i+1}" for i in range(len(df))]

        registro_seleccionado = st.selectbox(
            "Selecciona un registro para ver el detalle:",
            ["Selecciona"] + df["ID"].tolist()
        )

        if registro_seleccionado == "Selecciona":
            return

        registro = df[df["ID"] == registro_seleccionado].iloc[0].to_dict()
        st.subheader(f"🧾 {registro_seleccionado}")
        if st.session_state.get("voz_activa", False):
            leer_en_voz(f"Mostrando detalles del {registro_seleccionado}")

        # Cargar preguntas
        try:
            with open(RUTA_PREGUNTAS, encoding="utf-8") as f:
                preguntas = json.load(f)
        except FileNotFoundError:
            st.error("Error al cargar las preguntas de referencia.")
            return

        # Evaluar el modelo
        modelo_usado = None
        diagnostico = None

        if all(k in registro for k in ["Probabilidad Estimada 2", "Predicción Óptima 2"]):
            try:
                prob = float(registro["Probabilidad Estimada 2"])
                pred = int(registro["Predicción Óptima 2"])
                modelo = cargar_modelo2()
                modelo_usado = 2
            except Exception as e:
                st.warning(f"Error en datos del modelo avanzado: {e}")
                return
        elif all(k in registro for k in ["Probabilidad Estimada 1", "Predicción Óptima 1"]):
            try:
                prob = float(registro["Probabilidad Estimada 1"])
                pred = int(registro["Predicción Óptima 1"])
                modelo = cargar_modelo1()
                modelo_usado = 1
            except Exception as e:
                st.warning(f"Error en datos del modelo básico: {e}")
                return
        else:
            st.warning("Este registro no contiene datos de diagnóstico.")
            return

        # Resultado
        st.markdown("### 📊 Resultado de Evaluación")
        try:
            df_modelo = pd.DataFrame([{k: v for k, v in registro.items() if k in COLUMNAS_MODELO}])

            if df_modelo["sexo"].iloc[0] not in [1, 2]:
                df_modelo["sexo"] = df_modelo["sexo"].replace({"Hombre": 1, "Mujer": 2}).fillna(-1)

            X = df_modelo[COLUMNAS_MODELO].apply(pd.to_numeric, errors="coerce").fillna(-1)
            variables_relevantes = obtener_variables_importantes(modelo, X)
            mostrar_resultado_prediccion(pred, modelo_usado, variables_importantes=variables_relevantes)
        except Exception as e:
            st.error(f"Error al generar diagnóstico: {e}")
            if st.session_state.get("voz_activa", False):
                leer_en_voz("Ocurrió un error al procesar el resultado del participante.")

        # Mapeo de etiquetas
        mapeo_preguntas = {}
        for seccion in preguntas.values():
            if isinstance(seccion, list):
                for p in seccion:
                    if "codigo" in p:
                        mapeo_preguntas[p["codigo"]] = p.get("label", p["codigo"])
            elif isinstance(seccion, dict):
                for grupo in seccion.values():
                    for p in grupo:
                        if "codigo" in p:
                            mapeo_preguntas[p["codigo"]] = p.get("label", p["codigo"])

        # Respuestas
        st.markdown("### ✍🏽 Respuestas Registradas")
        for campo, valor in registro.items():
            if campo in ["Registrado por", "ID"] or pd.isna(valor):
                continue

            etiqueta = mapeo_preguntas.get(campo, campo.replace("_", " ").title())
            valor_mostrar = str(valor).strip()

            if campo == "sexo":
                valor_mostrar = "Hombre" if valor in [1, "1", "Hombre"] else "Mujer" if valor in [2, "2", "Mujer"] else valor_mostrar
            elif campo in ["Predicción Óptima 1", "Predicción Óptima 2"]:
                valor_mostrar = "Sí" if str(valor) == "1" else "No"

            st.markdown(f"**{etiqueta}:** {valor_mostrar}")

        # Descargar
        st.download_button(
            label="📥 Descargar informe completo",
            data=generar_pdf(
                [(mapeo_preguntas.get(k, k), str(v))
                 for k, v in registro.items()
                 if k not in ["Registrado por", "ID"]]
            ),
            file_name=f"Informe_{registro_seleccionado.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error al cargar registros: {e}")
        if st.session_state.get("voz_activa", False):
            leer_en_voz("Ocurrió un error al cargar los registros.")

def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "ubicacion_obtenida" not in st.session_state:
        st.session_state["ubicacion_obtenida"] = False
    if st.session_state["logged_in"]:
        st.sidebar.markdown("## Navegación")
        opcion = st.sidebar.radio("", ["Mi Cuenta", "Nuevo Registro", "Participante"])
        st.sidebar.button("🔴 Cerrar sesión", on_click=lambda: st.session_state.update({"logged_in": False, "usuario": None}))
        if opcion == "Mi Cuenta":
            mostrar_perfil()
        elif opcion == "Nuevo Registro":
            nuevo_registro()
        elif opcion == "Participante":
            mostrar_pacientes()
    else:
        login_page()

main()
