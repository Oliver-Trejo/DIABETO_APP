# APP con nuevo formulario
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

def conectar_google_sheet(nombre=None, key=None):
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = st.secrets["gcp_service_account"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(creds_dict), scope)
    client = gspread.authorize(creds)

    return client.open_by_key(key).sheet1 if key else client.open(nombre).sheet1


def render_pregunta(pregunta, key):
    tipo, label = pregunta["tipo"], pregunta["label"]
    if tipo == "text":
        return st.text_input(label, key=key)
    elif tipo == "number":
        return st.number_input(label, key=key)
    elif tipo == "textarea":
        return st.text_area(label, key=key)
    elif tipo == "select":
        opciones = ["Selecciona"] + pregunta["opciones"]
        seleccion = st.selectbox(label, opciones, key=key)
        if "valores" in pregunta and seleccion != "Selecciona":
            return pregunta["valores"][pregunta["opciones"].index(seleccion)]
        return "" if seleccion == "Selecciona" else seleccion

def obtener_variables_importantes(modelo, datos):
    # Buscar un paso del pipeline que tenga feature_importances_
    modelo_final = None
    if hasattr(modelo, "named_steps"):
        for name, step in modelo.named_steps.items():
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

    # Solo variables con valor "1"
    variables_marcadas = {col: val for col, val in fila.items() if str(val).strip() == "1"}

    variables_relevantes = []
    for i in top_indices:
        codigo = COLUMNAS_MODELO[i]
        if codigo in variables_marcadas:
            variables_relevantes.append((codigo, variables_marcadas[codigo]))
        if len(variables_relevantes) == 5:
            break

    return variables_relevantes


def generar_pdf(respuestas_completas, variables_relevantes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 10, "Respuestas del Paciente", ln=True)

    pdf.set_font("Arial", "", 10)
    for pregunta, respuesta in respuestas_completas:
        pregunta = str(pregunta).encode('latin-1', 'ignore').decode('latin-1')
        respuesta = str(respuesta).encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 10, f"{pregunta}: {respuesta}")

    pdf.add_page()
    pdf.set_font("Arial", "B", 10)
    pdf.cell(0, 10, "Preguntas Más Relevantes", ln=True)
    
    pdf.set_font("Arial", "", 10)
    for pregunta, respuesta in variables_relevantes:
        pregunta = str(pregunta).encode('latin-1', 'ignore').decode('latin-1')
        respuesta = str(respuesta).encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 10, f"{pregunta}: {respuesta}")

    buffer = BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin-1')  # Genera contenido como string y codifica
    buffer = BytesIO(pdf_bytes)
    buffer.seek(0)
    return buffer

def obtener_hoja_usuarios():
    return conectar_google_sheet(key=st.secrets["google_sheets"]["usuarios_key"])

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

    st.markdown("""<div class='form-container'><div style='text-align:center; margin-bottom:25px;'>
    <h1 style='color:black;'>DIABETO<br>Queremos ayudarte a saber si tienes señales que podrían indicar riesgo de diabetes tipo 2. Es rápido y fácil.</h1></div>""", unsafe_allow_html=True)

    # Activar modo de voz
    if "voz_activa" not in st.session_state:
        st.session_state["voz_activa"] = False

    st.session_state["voz_activa"] = st.checkbox("🗣️ ¿Deseas activar el modo de lectura en voz alta?", value=st.session_state["voz_activa"])

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
                usuario = buscar_usuario_por_nombre(nombre)
                if usuario and usuario["Contraseña Hasheada"] == hash_password(password):
                    st.session_state["logged_in"] = True
                    st.session_state["usuario"] = nombre
                    st.sidebar.markdown(f"👤 Sesión activa: **{st.session_state['usuario']}**")
                    st.success(f"Bienvenido, {nombre}")
                    if st.session_state["voz_activa"]:
                        leer_en_voz(f"Bienvenido, {nombre}. Has iniciado sesión correctamente.")
                    st.rerun()
                else:
                    st.error("No pudimos encontrar tus datos. Revisa que estén bien escritos o intenta registrarte.")
                    if st.session_state["voz_activa"]:
                        leer_en_voz("No pudimos encontrar tus datos. Intenta de nuevo o crea una cuenta.")

    elif modo == "Crear cuenta":
        if st.session_state["voz_activa"]:
            leer_en_voz("Por favor, escribe tu nombre completo y una contraseña para crear una cuenta nueva.")

        with st.form("registro_form"):
            nombre = st.text_input("Nombre completo", key="reg_nombre")
            password = st.text_input("Contraseña", type="password", key="reg_pass")
            if st.form_submit_button("Registrar"):
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

    st.markdown("</div>", unsafe_allow_html=True)

def mostrar_perfil():
    st.title("👩🏽👨🏽 Mi Cuenta")
    
    # Estilo para agrandar solo el texto de introducción
    st.markdown("""
        <style>
            .perfil-container {
                text-align: center;
                margin-top: 35px;
            }
            .texto-introductorio {
                font-size: 22px;
                line-height: 1.6;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='perfil-container'>", unsafe_allow_html=True)

    # Obtener y mostrar nombre del usuario
    nombre_usuario = st.session_state.get("usuario", "Usuario")
    st.markdown(f"<h2>{nombre_usuario}</h2>", unsafe_allow_html=True)

    # 🔊 Leer nombre del usuario si el modo voz está activado
    if st.session_state.get("voz_activa", False):
        leer_en_voz(f"Bienvenido, {nombre_usuario}. Esta es tu cuenta.")

    try:
        with open("intro_text.json", encoding="utf-8") as f:
            textos = json.load(f)
            texto_crudo = textos.get("mi_cuenta", "")

            # Reemplazo simple de Markdown por HTML
            texto_html = (
                texto_crudo.replace("**", "<b>")
                           .replace("*", "<i>")
                           .replace("<b><i>", "<b><i>")
                           .replace("</i></b>", "</i></b>")
            )

            # Mostrar con estilo
            st.markdown(f"<div class='texto-introductorio'>{texto_html}</div>", unsafe_allow_html=True)

            # 🔊 Leer también el texto introductorio si está activado
            if st.session_state.get("voz_activa", False) and texto_crudo:
                texto_sin_html = re.sub(r'<[^>]+>', '', texto_crudo)  # Elimina todas las etiquetas HTML
                leer_en_voz(texto_sin_html.strip())

    except FileNotFoundError:
        st.warning("No se encontró el archivo de texto introductorio.")
        if st.session_state.get("voz_activa", False):
            leer_en_voz("No se encontró el texto introductorio.")

    st.markdown("</div>", unsafe_allow_html=True)


def mostrar_pacientes():
    st.title("📋 Participante")

    # Configuración inicial
    if st.session_state.get("voz_activa", False):
        leer_en_voz("Estás en la sección de participantes. Aquí puedes consultar los registros guardados.")

    try:
        # Cargar datos desde Google Sheets
        sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
        registros = sheet.get_all_records()
        
        if not registros:
            st.info("No hay registros disponibles en la base de datos.")
            return
            
        df = pd.DataFrame(registros).dropna(how="all")
        
        # Filtrar por usuario actual
        usuario_actual = st.session_state.get("usuario", "").strip().lower()
        df = df[df["Registrado por"].str.strip().str.lower() == usuario_actual]

        if df.empty:
            st.info("No tienes registros guardados aún. Crea uno en 'Nuevo Registro'.")
            return

        # Generar IDs legibles
        df["ID"] = [f"Registro #{i+1}" for i in range(len(df))]
        
        # Selección de registro
        registro_seleccionado = st.selectbox(
            "Selecciona un registro para ver el detalle:", 
            ["Selecciona"] + df["ID"].tolist()
        )

        if registro_seleccionado == "Selecciona":
            return

        # Obtener registro específico
        registro = df[df["ID"] == registro_seleccionado].iloc[0].to_dict()
        
        # Mostrar encabezado
        st.subheader(f"🧾 {registro_seleccionado}")
        if st.session_state.get("voz_activa", False):
            leer_en_voz(f"Mostrando detalles del {registro_seleccionado}")

        # Cargar estructura de preguntas
        try:
            with open(RUTA_PREGUNTAS, encoding="utf-8") as f:
                preguntas = json.load(f)
        except FileNotFoundError:
            st.error("Error al cargar las preguntas de referencia")
            return

        # Procesamiento de diagnóstico
        modelo_usado = None
        diagnostico = None
        
        # Verificar modelo 2 primero
        if all(k in registro for k in ["Probabilidad Estimada 2", "Predicción Óptima 2"]):
            try:
                prob = float(registro["Probabilidad Estimada 2"])
                pred = int(registro["Predicción Óptima 2"])
                modelo = cargar_modelo2()
                modelo_usado = 2
            except (ValueError, TypeError) as e:
                st.warning(f"Datos inválidos en predicción avanzada: {str(e)}")
                return
        # Verificar modelo 1
        elif all(k in registro for k in ["Probabilidad Estimada 1", "Predicción Óptima 1"]):
            try:
                prob = float(registro["Probabilidad Estimada 1"])
                pred = int(registro["Predicción Óptima 1"])
                modelo = cargar_modelo1()
                modelo_usado = 1
            except (ValueError, TypeError) as e:
                st.warning(f"Datos inválidos en predicción inicial: {str(e)}")
                return
        else:
            st.warning("Este registro no contiene datos de diagnóstico completos")
            return

        # Mostrar resultado
        with st.container():
            st.markdown("### 📊 Resultado de Evaluación")
            
            # Obtener variables importantes
            df_modelo = pd.DataFrame([registro])
            df_modelo["sexo"] = df_modelo["sexo"].replace({"Hombre": 1, "Mujer": 2})
            X = df_modelo[COLUMNAS_MODELO].fillna(-1).astype(float)
            
            try:
                variables_relevantes = obtener_variables_importantes(modelo, X)
                diagnostico = mostrar_resultado_prediccion(
                    pred=pred,
                    modelo_usado=modelo_usado,
                    variables_importantes=variables_relevantes
                )
            except Exception as e:
                st.error(f"Error al generar diagnóstico: {str(e)}")

        # Mostrar respuestas detalladas
        st.markdown("### ✍🏽 Respuestas Registradas")
        
        # Mapeo de códigos a preguntas
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

        # Mostrar respuestas organizadas
        for campo, valor in registro.items():
            if campo in ["Registrado por", "ID"] or pd.isna(valor):
                continue
                
            etiqueta = mapeo_preguntas.get(campo, campo.replace("_", " ").title())
            valor_mostrar = str(valor).strip()
            
            # Formatear valores especiales
            if campo in ["sexo"]:
                valor_mostrar = "Hombre" if valor == 1 else "Mujer" if valor == 2 else valor_mostrar
            elif campo in ["Predicción Óptima 1", "Predicción Óptima 2"]:
                valor_mostrar = "Sí" if int(float(valor)) == 1 else "No"
            
            st.markdown(f"**{etiqueta}:** {valor_mostrar}")

        # Botón de descarga
        st.download_button(
            label="📥 Descargar informe completo",
            data=generar_pdf(
                [(mapeo_preguntas.get(k, k), str(v))  # <-- Ahora con paréntesis cerrado
                for k, v in registro.items() 
                if k not in ["Registrado por", "ID"]]
            ),
            file_name=f"Informe_{registro_seleccionado.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Error al cargar los registros: {str(e)}")
        if st.session_state.get("voz_activa", False):
            leer_en_voz("Ocurrió un error al cargar los registros. Por favor intenta nuevamente.")


def guardar_respuesta_paciente(fila_dict):
    """
    Guarda un registro de paciente en Google Sheets con validación y manejo de errores
    
    Args:
        fila_dict (dict): Diccionario con los datos del paciente a guardar
        
    Returns:
        bool: True si se guardó correctamente, False si hubo error
    """
    MAX_INTENTOS = 3
    intentos = 0
    
    while intentos < MAX_INTENTOS:
        try:
            # 1. Validación y limpieza inicial
            if not fila_dict or not isinstance(fila_dict, dict):
                st.error("Datos de paciente inválidos")
                return False
                
            # Limpieza de valores
            fila_limpia = {
                k: str(v).strip() if v is not None else ""
                for k, v in fila_dict.items()
                if v not in [None, "", " "]  # Filtrar valores vacíos
            }
            
            # Campos obligatorios
            campos_requeridos = [
                "Registrado por",
                "Fecha",
                "Probabilidad Estimada 1",
                "Predicción Óptima 1"
            ]
            
            for campo in campos_requeridos:
                if campo not in fila_limpia:
                    st.warning(f"Falta campo requerido: {campo}")
                    fila_limpia[campo] = "N/A"  # Valor por defecto

            # 2. Conexión con Google Sheets
            sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
            encabezados = sheet.row_values(1)
            
            if not encabezados:  # Si la hoja está vacía
                encabezados = list(fila_limpia.keys())
                sheet.append_row(encabezados)  # Crear encabezados
            
            # 3. Verificar y actualizar columnas faltantes
            nuevos_campos = [campo for campo in fila_limpia.keys() if campo not in encabezados]
            
            if nuevos_campos:
                # Actualizar encabezados en lote
                sheet.insert_row(nuevos_campos, 2)
                encabezados.extend(nuevos_campos)
                
            # 4. Preparar fila en el orden correcto
            fila_ordenada = []
            for columna in encabezados:
                valor = fila_limpia.get(columna, "")
                
                # Conversión especial para campos numéricos
                if columna in ["Probabilidad Estimada 1", "Probabilidad Estimada 2"]:
                    try:
                        valor = f"{float(valor):.4f}" if valor else ""
                    except:
                        valor = ""
                elif columna in ["Predicción Óptima 1", "Predicción Óptima 2"]:
                    valor = str(int(float(valor))) if valor else ""
                
                fila_ordenada.append(valor)

            # 5. Guardar datos
            sheet.append_row(fila_ordenada)
            
            # 6. Validar que se guardó correctamente
            ultima_fila = sheet.get_all_records()[-1]
            if str(ultima_fila.get("Registrado por", "")) != str(fila_limpia.get("Registrado por", "")):
                raise ValueError("Error de verificación al guardar")
                
            st.toast("Datos guardados correctamente", icon="✅")
            return True
            
        except gspread.exceptions.APIError as e:
            intentos += 1
            if intentos >= MAX_INTENTOS:
                st.error(f"Error al conectar con Google Sheets (intento {intentos}/{MAX_INTENTOS}): {str(e)}")
                return False
            time.sleep(2)  # Espera antes de reintentar
            
        except Exception as e:
            st.error(f"Error inesperado al guardar: {str(e)}")
            return False
    
    return False

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
    try:
        sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
        encabezados = sheet.row_values(1)
        
        # Asegurar columnas básicas
        campos_requeridos = [
            "Registrado por",
            "Probabilidad Estimada 1",
            "Predicción Óptima 1",
            "Probabilidad Estimada 2",
            "Predicción Óptima 2"
        ]
        
        # Verificar y añadir campos faltantes
        nuevos_encabezados = []
        for campo in campos_requeridos + list(fila_dict.keys()):
            if campo not in encabezados and campo not in nuevos_encabezados:
                nuevos_encabezados.append(campo)
        
        if nuevos_encabezados:
            sheet.insert_row(nuevos_encabezados, 2)  # Añadir nuevas columnas
        
        # Preparar fila con todos los campos
        fila_completa = []
        for col in sheet.row_values(1):  # Usar encabezados actualizados
            if col in fila_dict:
                # Convertir valores a string y limpiar
                valor = str(fila_dict[col]).strip()
                # Manejar valores booleanos
                if valor.lower() == 'true':
                    valor = '1'
                elif valor.lower() == 'false':
                    valor = '0'
                fila_completa.append(valor)
            else:
                fila_completa.append('')
        
        # Añadir fila
        sheet.append_row(fila_completa)
        st.toast("Datos guardados correctamente", icon="✅")
        return True
        
    except Exception as e:
        st.error(f"Error al guardar: {str(e)}")
        if st.session_state.get("voz_activa", False):
            leer_en_voz("Error al guardar los datos. Por favor intente nuevamente.")
        return False

def mostrar_resultado_prediccion(pred, modelo_usado, variables_importantes=None):
    """
    Muestra el resultado de la predicción sin porcentajes y con la lógica corregida para determinar el modelo usado.
    
    Args:
        pred (int): Predicción (0 o 1)
        modelo_usado (int): 1 para modelo inicial, 2 para modelo secundario
        variables_importantes (list): Lista de tuplas con variables relevantes
    """
    # Determinar diagnóstico según el modelo usado
    if modelo_usado == 2:
        diagnostico = "Prediabético" if pred == 0 else "Diabético"
        color = "#FFA500" if pred == 0 else "#FF0000"  # Naranja para prediabetes, rojo para diabetes
        emoji = "🟠" if pred == 0 else "🚨"
        mensaje = (
            "Tus respuestas indican señales compatibles con una condición prediabética. "
            "Te recomendamos consultar a un especialista para una evaluación más detallada."
            if pred == 0 else
            "Tus respuestas indican señales compatibles con diabetes tipo 2. "
            "Es importante que acudas a un centro de salud para una evaluación médica."
        )
    else:
        diagnostico = "Sano" if pred == 0 else "En Riesgo"
        color = "#4CAF50" if pred == 0 else "#FFA500"  # Verde para sano, naranja para riesgo
        emoji = "✅" if pred == 0 else "⚠️"
        mensaje = (
            "¡Buenas noticias! No encontramos señales claras de diabetes. "
            "Mantén hábitos saludables para prevenir."
            if pred == 0 else
            "Tus respuestas muestran factores de riesgo. Continuaremos con una evaluación más detallada."
        )

    # Mostrar resultado en la interfaz
    st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; border-left: 5px solid {color}; margin-bottom:20px;'>
            <h3 style='color:{color}; margin-top:0;'>{emoji} Diagnóstico: {diagnostico}</h3>
            <p style='margin-bottom:0;'>{mensaje}</p>
        </div>
    """, unsafe_allow_html=True)

    # Mostrar variables importantes si es relevante
    texto_a_leer = mensaje
    if pred == 1 and variables_importantes:
        st.markdown("#### 🔍 Factores más relevantes en esta evaluación:")
        texto_a_leer += " Los factores más relevantes fueron: "
        
        for var, val in variables_importantes:
            st.markdown(f"- **{var}**: {val}")
            texto_a_leer += f"{var}, "

    # Lectura en voz alta si está activado
    if st.session_state.get("voz_activa", False):
        leer_en_voz(texto_a_leer)

    return diagnostico


def ejecutar_prediccion():
    sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
    df = pd.DataFrame(sheet.get_all_records())
    if df.empty:
        st.warning("No hay datos suficientes para predecir.")
        return
    faltantes = [col for col in COLUMNAS_MODELO if col not in df.columns]
    if faltantes:
        st.error(f"Faltan columnas: {faltantes}")
        return
    X = df.iloc[[-1]][COLUMNAS_MODELO].replace("", -1)
    modelo = cargar_modelo2()
    proba = modelo.predict_proba(X)[0, 1]
    pred = int(proba >= 0.21)
    mostrar_resultado_prediccion(proba, pred)

def nuevo_registro():
    st.title("📝 Registro de Pacientes")
    
    # Configuración inicial
    if st.session_state.get("voz_activa", False):
        leer_en_voz("Estás en la sección de registro de pacientes. Por favor responde las siguientes preguntas.")

    try:
        # Cargar estructura del formulario
        with open(RUTA_PREGUNTAS, encoding="utf-8") as f:
            secciones = json.load(f)
    except FileNotFoundError:
        st.error("Error crítico: No se encontró el archivo de preguntas. Contacta al administrador.")
        return
    except json.JSONDecodeError:
        st.error("Error: Archivo de preguntas mal formado")
        return

    # Inicializar respuestas
    respuestas = {}
    key_form = f"form_registro_{st.session_state.get('usuario', 'anon')}_{int(time.time())}"

    # Mostrar formulario
    with st.form(key=key_form, clear_on_submit=True):
        # Renderizar todas las preguntas organizadas
        for titulo, contenido in secciones.items():
            st.subheader(titulo)
            
            if st.session_state.get("voz_activa", False):
                leer_en_voz(titulo)

            if titulo == "Familia":
                for familiar, grupo in contenido.items():
                    with st.expander(f"Antecedentes familiares: {familiar}"):
                        for p in grupo:
                            codigo = p.get("codigo", f"{p['label']}_{uuid.uuid4().hex[:6]}")
                            respuestas[codigo] = render_pregunta(p, key=f"{key_form}_{codigo}")
            else:
                for p in contenido:
                    codigo = p.get("codigo", f"{p['label']}_{uuid.uuid4().hex[:6]}")
                    respuestas[codigo] = render_pregunta(p, key=f"{key_form}_{codigo}")

        # Botón de envío con validación
        submitted = st.form_submit_button("💾 Guardar y Evaluar",
                                         help="Guarda las respuestas y realiza la evaluación de riesgo")
        
        if submitted:
            with st.spinner("Analizando respuestas..."):
                try:
                    # 1. Validación básica de campos obligatorios
                    campos_requeridos = ['edad', 'sexo', 'peso', 'talla']
                    faltantes = [campo for campo in campos_requeridos if campo not in respuestas or not respuestas[campo]]
                    if faltantes:
                        raise ValueError(f"Campos obligatorios faltantes: {', '.join(faltantes)}")

                    # 2. Convertir a DataFrame y limpiar datos
                    df_modelo = pd.DataFrame([respuestas])
                    
                    # 3. Realizar predicción
                    resultado = predecir_nuevos_registros(df_modelo)
                    
                    if resultado is None:
                        raise RuntimeError("No se pudo completar la evaluación")

                    # 4. Determinar modelo usado y predicción
                    if "Predicción Óptima 2" in resultado.columns and not pd.isna(resultado["Predicción Óptima 2"].iloc[0]):
                        modelo_usado = 2
                        pred = int(resultado["Predicción Óptima 2"].iloc[0])
                    else:
                        modelo_usado = 1
                        pred = int(resultado["Predicción Óptima 1"].iloc[0])

                    # 5. Obtener variables importantes
                    modelo = cargar_modelo2() if modelo_usado == 2 else cargar_modelo1()
                    variables_relevantes = obtener_variables_importantes(modelo, resultado)

                    # 6. Guardar en Google Sheets
                    registro_completo = resultado.iloc[0].to_dict()
                    registro_completo["Registrado por"] = st.session_state.get("usuario", "Anónimo")
                    registro_completo["Fecha"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    if not guardar_respuesta_paciente(registro_completo):
                        raise RuntimeError("Error al guardar en la base de datos")

                    # 7. Mostrar resultados
                    st.success("✅ Evaluación completada y guardada")
                    
                    # Mostrar diagnóstico
                    diagnostico = mostrar_resultado_prediccion(
                        pred=pred,
                        modelo_usado=modelo_usado,
                        variables_importantes=variables_relevantes
                    )

                    # Generar PDF con resultados
                    with st.expander("📄 Descargar resumen", expanded=False):
                        # Preparar datos para PDF
                        datos_pdf = []
                        for seccion in secciones.values():
                            if isinstance(seccion, list):
                                for p in seccion:
                                    if 'codigo' in p and 'label' in p:
                                        valor = respuestas.get(p['codigo'], 'No respondido')
                                        datos_pdf.append((p['label'], str(valor)))
                            elif isinstance(seccion, dict):
                                for grupo in seccion.values():
                                    for p in grupo:
                                        if 'codigo' in p and 'label' in p:
                                            valor = respuestas.get(p['codigo'], 'No respondido')
                                            datos_pdf.append((p['label'], str(valor)))

                        pdf_buffer = generar_pdf(datos_pdf, variables_relevantes)
                        
                        st.download_button(
                            label="⬇️ Descargar Informe Completo",
                            data=pdf_buffer,
                            file_name=f"Evaluación_DIABETO_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )

                except ValueError as e:
                    st.error(f"Datos incompletos o inválidos: {str(e)}")
                except Exception as e:
                    st.error(f"Error inesperado: {str(e)}")
                    if st.session_state.get("voz_activa", False):
                        leer_en_voz("Ocurrió un error al procesar tus respuestas. Por favor intenta nuevamente.")


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
