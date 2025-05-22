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

    if st.session_state.get("voz_activa", False):
        leer_en_voz("Estás en la sección de participantes. Aquí puedes consultar los registros guardados.")

    sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
    df = pd.DataFrame(sheet.get_all_records())

    usuario = st.session_state.get("usuario", "").strip().lower()
    df = df[df["Registrado por"].str.strip().str.lower() == usuario]

    if df.empty:
        st.info("Todavía no hay ningún registro guardado. Puedes crear uno en la sección de ‘Nuevo Registro’.")
        if st.session_state.get("voz_activa", False):
            leer_en_voz("Todavía no tienes ningún registro guardado. Ve a la sección de nuevo registro para crear uno.")
        return

    df = df.dropna(how="all").reset_index(drop=True)
    df["ID Paciente"] = ["Registro #" + str(i + 1) for i in df.index]
    seleccionado = st.selectbox("Selecciona un registro para ver el detalle:", ["Selecciona"] + df["ID Paciente"].tolist())

    if seleccionado == "Selecciona":
        return

    if st.session_state.get("voz_activa", False):
        leer_en_voz(f"Has seleccionado el {seleccionado}. Mostrando los detalles.")

    idx = df[df["ID Paciente"] == seleccionado].index[0]
    registro = df.iloc[idx]
    st.subheader(f"🧾 {seleccionado}")

    with open(RUTA_PREGUNTAS, encoding="utf-8") as f:
        preguntas_json = json.load(f)

    codigo_a_label = {}
    codigo_a_opciones = {}

    for bloque in ["Generales", "Familia", "Hábitos"]:
        contenido = preguntas_json.get(bloque, {})
        if isinstance(contenido, list):
            for p in contenido:
                codigo = p.get("codigo")
                if codigo:
                    codigo_a_label[codigo] = p.get("label", codigo)
                    if "valores" in p and "opciones" in p:
                        codigo_a_opciones[codigo] = dict(zip(p["valores"], p["opciones"]))
        elif isinstance(contenido, dict):
            for grupo in contenido.values():
                for p in grupo:
                    codigo = p.get("codigo")
                    if codigo:
                        codigo_a_label[codigo] = p.get("label", codigo)
                        if "valores" in p and "opciones" in p:
                            codigo_a_opciones[codigo] = dict(zip(p["valores"], p["opciones"]))

    for familiar, grupo in preguntas_json.get("Antecedentes familiares", {}).items():
        for p in grupo:
            codigo = p.get("codigo")
            if codigo:
                codigo_a_label[codigo] = p.get("label", codigo)
                if "valores" in p and "opciones" in p:
                    codigo_a_opciones[codigo] = dict(zip(p["valores"], p["opciones"]))

    # 🔍 Determinar qué modelo se usó
    variables_etiquetadas = []
    if "Probabilidad Estimada 2" in registro and "Predicción Óptima 2" in registro:
        try:
            prob = float(registro["Probabilidad Estimada 2"])
            pred = int(registro["Predicción Óptima 2"])
            modelo = cargar_modelo2()
        except (ValueError, TypeError):
            st.warning("⚠️ Este registro tiene valores inválidos en la predicción 2.")
            return
    elif "Probabilidad Estimada 1" in registro and "Predicción Óptima 1" in registro:
        try:
            prob = float(registro["Probabilidad Estimada 1"])
            pred = int(registro["Predicción Óptima 1"])
            modelo = cargar_modelo1()
        except (ValueError, TypeError):
            st.warning("⚠️ Este registro tiene valores inválidos en la predicción 1.")
            return
    else:
        st.warning("No hay predicción guardada para este registro.")
        return

    df_modelo = registro.to_frame().T
    df_modelo["sexo"] = df_modelo["sexo"].replace({"Hombre": 1, "Mujer": 2})
    X = df_modelo[COLUMNAS_MODELO].replace("", -1).astype(float)
    df_modelo['Probabilidad Estimada'] = modelo.predict_proba(X)[:, 1]
    df_modelo['Predicción Óptima'] = (df_modelo['Probabilidad Estimada'] >= 0.18).astype(int)

    variables_relevantes = obtener_variables_importantes(modelo, df_modelo)
    for var, val in variables_relevantes:
        nombre = codigo_a_label.get(var, var)
        if var in codigo_a_opciones:
            try:
                val = codigo_a_opciones[var].get(int(val), val)
            except:
                pass
        variables_etiquetadas.append((nombre, val))

    texto_diagnostico = mostrar_resultado_prediccion(prob, pred, variables_etiquetadas)
    if st.session_state.get("voz_activa", False):
        leer_en_voz(texto_diagnostico)

    st.markdown("#### ✍🏽 Tus respuestas")
    respuestas_mostradas = []
    for campo, valor in registro.items():
        if campo in ["Registrado por", "ID Paciente"]:
            continue
        label = codigo_a_label.get(campo, campo)
        if campo in codigo_a_opciones:
            try:
                valor = codigo_a_opciones[campo].get(int(valor), valor)
            except:
                pass
        respuestas_mostradas.append((label, valor))
        st.markdown(f"**{label}:** {valor}")
        if st.session_state.get("voz_activa", False):
            leer_en_voz(f"{label}: {valor}")

    if st.session_state.get("voz_activa", False):
        leer_en_voz("Presiona el botón azul con rojo de la parte de abajo para descargar tus respuestas.")

    if st.button("📥 Descargar resumen de respuestas"):
        pdf_buffer = generar_pdf(respuestas_mostradas, variables_etiquetadas)
        st.download_button("Descargar respuestas en PDF", data=pdf_buffer, file_name=f"{seleccionado}.pdf", mime="application/pdf")




def predecir_nuevos_registros(df_input, threshold1=0.18, threshold2=0.18):
    modelo1 = cargar_modelo1()
    X1 = df_input[COLUMNAS_MODELO].replace("", -1).astype(float)
    df_input["Probabilidad Estimada 1"] = modelo1.predict_proba(X1)[:, 1]
    df_input["Predicción Óptima 1"] = (df_input["Probabilidad Estimada 1"] >= threshold1).astype(int)

    if df_input["Predicción Óptima 1"].iloc[0] == 1:
        modelo2 = cargar_modelo2()
        X2 = df_input[COLUMNAS_MODELO].replace("", -1).astype(float)
        df_input["Probabilidad Estimada 2"] = modelo2.predict_proba(X2)[:, 1]
        df_input["Predicción Óptima 2"] = (df_input["Probabilidad Estimada 2"] >= threshold2).astype(int)

    return df_input


def guardar_respuesta_paciente(fila_dict, proba=None, pred=None):
    sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
    encabezados = sheet.row_values(1)

    # Añadir campos principales
    fila_dict["Registrado por"] = st.session_state.get("usuario", "Desconocido")

    # Añadir columnas del modelo 1 si no existen
    if "Probabilidad Estimada 1" not in fila_dict:
        fila_dict["Probabilidad Estimada 1"] = ""
    if "Predicción Óptima 1" not in fila_dict:
        fila_dict["Predicción Óptima 1"] = ""

    # Añadir columnas del modelo 2 si existen
    if "Probabilidad Estimada 2" in fila_dict or "Predicción Óptima 2" in fila_dict:
        if "Probabilidad Estimada 2" not in encabezados:
            sheet.update_cell(1, len(encabezados) + 1, "Probabilidad Estimada 2")
            encabezados.append("Probabilidad Estimada 2")
        if "Predicción Óptima 2" not in encabezados:
            sheet.update_cell(1, len(encabezados) + 1, "Predicción Óptima 2")
            encabezados.append("Predicción Óptima 2")

    # Verificar que encabezados estén actualizados
    for clave in fila_dict.keys():
        if clave not in encabezados:
            sheet.update_cell(1, len(encabezados) + 1, clave)
            encabezados.append(clave)

    # Crear la nueva fila respetando el orden de encabezados
    nueva_fila = [fila_dict.get(col, "") for col in encabezados]
    sheet.append_row(nueva_fila)


def mostrar_resultado_prediccion(proba, pred, variables_importantes=None):
    # Determinar tipo de salida según rango de probabilidad
    if "Probabilidad Estimada 2" in st.session_state:
        # Modelo 2: ya se activó y pred viene de ahí
        diagnostico = "Prediabético" if pred == 0 else "Diabético"
        color = "#FFA500" if pred == 0 else "#FF0000"
        emoji = "🟠" if pred == 0 else "🚨"
        mensaje = (
            "Tus respuestas indican señales compatibles con una condición prediabética."
            if pred == 0 else
            "Tus respuestas indican señales compatibles con diabetes tipo 2. Te recomendamos acudir a un centro de salud."
        )
    else:
        # Modelo 1: diagnóstico inicial
        diagnostico = "Sano" if pred == 0 else "En Riesgo"
        color = "#4CAF50" if pred == 0 else "#FFA500"
        emoji = "✅" if pred == 0 else "⚠️"
        mensaje = (
            "¡Buenas noticias! No encontramos señales claras de diabetes. Aun así, cuida tu salud."
            if pred == 0 else
            "Tus respuestas son similares a las de personas con diabetes. Continuaremos con una evaluación más detallada."
        )

    st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; border-left: 5px solid {color};'>
            <h3 style='color:{color};'>{emoji} Diagnóstico: {diagnostico}</h3>
            <p>{mensaje}</p>
            <p style='font-weight:bold;'>Tu perfil coincide con personas que tienen diabetes en un: {proba:.2%}</p>
        </div>
    """, unsafe_allow_html=True)

    texto_a_leer = f"{mensaje}. Tu perfil coincide con personas con diabetes en un {proba:.0f} por ciento. "

    if pred == 1 and variables_importantes:
        st.markdown("#### 🔍 Las siguientes respuestas fueron importantes para este resultado:")
        texto_a_leer += "Las siguientes respuestas fueron importantes para este resultado. "

        for var, val in variables_importantes:
            st.markdown(f"- **{var}**: {val}")
            texto_a_leer += f"{var}: {val}. "

    if st.session_state.get("voz_activa", False):
        leer_en_voz(texto_a_leer)

    return texto_a_leer


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

    if st.session_state.get("voz_activa", False):
        leer_en_voz("Estás en la sección de registro de pacientes. Por favor responde las siguientes preguntas.")

    # ✅ Cargar preguntas del formulario
    with open(RUTA_PREGUNTAS, encoding="utf-8") as f:
        secciones = json.load(f)

    respuestas = {}

    if st.session_state.get("mostrar_prediccion"):
        ejecutar_prediccion()
        st.session_state["mostrar_prediccion"] = False

    key_form = "formulario_registro_" + st.session_state.get("usuario", str(uuid.uuid4()))

    with st.form(key=key_form):
        for titulo, preguntas in secciones.items():
            st.subheader(titulo)
            if st.session_state.get("voz_activa", False):
                leer_en_voz(f"Sección: {titulo}")

            if titulo == "Familia":
                for familiar, grupo in preguntas.items():
                    st.markdown(f"### {familiar}")
                    if st.session_state.get("voz_activa", False):
                        leer_en_voz(f"{familiar}")
                    for i, p in enumerate(grupo):
                        codigo = p.get("codigo", f"{p['label']}-{i}")
                        if st.session_state.get("voz_activa", False):
                            leer_en_voz(p.get("label", ""))
                        respuestas[codigo] = render_pregunta(p, key=codigo)
            else:
                for i, p in enumerate(preguntas):
                    codigo = p.get("codigo", f"{p['label']}-{i}")
                    if st.session_state.get("voz_activa", False):
                        leer_en_voz(p.get("label", ""))
                    respuestas[codigo] = render_pregunta(p, key=codigo)

        if st.form_submit_button("Guardar"):
            # 🔹 Hacer predicción con modelo 1 (y luego con modelo 2 si aplica)
            df_modelo = pd.DataFrame([respuestas])
            resultado = predecir_nuevos_registros(df_modelo)

            if "Predicción Óptima 2" in resultado.columns:
                proba = resultado["Probabilidad Estimada 2"].iloc[0]
                pred = resultado["Predicción Óptima 2"].iloc[0]
            else:
                proba = resultado["Probabilidad Estimada 1"].iloc[0]
                pred = resultado["Predicción Óptima 1"].iloc[0]

            # 🔹 Guardar paciente
            guardar_respuesta_paciente(respuestas, proba, pred)

            st.success("✅ Registro guardado correctamente.")
            if st.session_state.get("voz_activa", False):
                leer_en_voz("Registro guardado correctamente. Mostrando resultados.")
            st.session_state["mostrar_prediccion"] = True

            modelo_usado = cargar_modelo2() if pred == 1 and "Predicción Óptima 2" in resultado.columns else cargar_modelo1()
            variables_relevantes = obtener_variables_importantes(modelo_usado, df_modelo)

            mostrar_resultado_prediccion(proba, pred, variables_relevantes)
            st.rerun()


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
