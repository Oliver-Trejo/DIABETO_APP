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
import os

# --- CONFIGURACIONES GLOBALES ---
st.set_page_config(page_title="DIABETO", page_icon="🏥", layout="wide")
RUTA_PREGUNTAS = "preguntas2.json"
COLUMNAS_MODELO = ["sexo", "edad", "a0201", "a0206", "a0601",
    "a0602a", "a0602b", "a0602c", "a0602d", "a0701a", "a0701b", "a0703", "a0704",
    "a0801a", "a0803a", "a0804a", "a0806a", "a0801b", "a0803b", "a0804b", "a0806b",
    "a0801c", "a0803c", "a0804c", "a0806c", "a1401", "a1405",
    "peso", "talla", "cintura"
]

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
    tipo = pregunta.get("tipo", "text")
    label = pregunta.get("label", "Pregunta sin título")
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

def predecir_nuevos_registros(df_input, threshold1=0.33, threshold2=0.49):
    modelo1 = cargar_modelo1()

    # Validar y limpiar datos de entrada
    X = df_input[COLUMNAS_MODELO].copy()

    # Reemplazar vacíos con -1 y convertir a float de manera segura
    for col in COLUMNAS_MODELO:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1)

    # Predicción con Modelo 1
    df_input["Probabilidad Estimada 1"] = modelo1.predict_proba(X)[:, 1]
    df_input["Predicción Óptima 1"] = (df_input["Probabilidad Estimada 1"] >= threshold1).astype(int)

    # Si el resultado fue positivo, aplicar Modelo 2
    if df_input["Predicción Óptima 1"].iloc[0] == 1:
        modelo2 = cargar_modelo2()
        df_input["Probabilidad Estimada 2"] = modelo2.predict_proba(X)[:, 1]
        df_input["Predicción Óptima 2"] = (df_input["Probabilidad Estimada 2"] >= threshold2).astype(int)
    else:
        df_input["Probabilidad Estimada 2"] = None
        df_input["Predicción Óptima 2"] = None

    return df_input

def guardar_respuesta_paciente(fila_dict):
    sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
    encabezados = sheet.row_values(1)

    fila_dict["Registrado por"] = st.session_state.get("usuario", "Desconocido")

    nuevos = [k for k in fila_dict.keys() if k not in encabezados]
    if nuevos:
        encabezados += nuevos
        sheet.delete_row(1)
        sheet.insert_row(encabezados, 1)

    nueva_fila = [fila_dict.get(col, "") for col in encabezados]
    sheet.append_row(nueva_fila)

def analizar_diagnostico(fila):
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    try:
        pred1 = int(fila.get("Predicción Óptima 1", 0))
        prob1 = safe_float(fila.get("Probabilidad Estimada 1"))
        pred2 = fila.get("Predicción Óptima 2", "")
        prob2 = safe_float(fila.get("Probabilidad Estimada 2"))

        if pred1 == 0:
            return "Perfil Sano", "¡Buenas noticias! No encontramos señales claras de Diabetes. Aun así, cuida tu salud.", "✅", "#4CAF50", prob1
        elif str(pred2) == "0":
            return "Perfil Prediabético", "Tus respuestas indican señales compatibles con una condición Prediabética. Te recomendamos consultar a un especialista.", "🟠", "#FFA500", prob2
        elif str(pred2) == "1":
            return "Perfil Diabético", "Tus respuestas indican señales compatibles con Diabetes Tipo 2. Es importante que acudas a un centro de salud lo antes posible.", "🚨", "#FF0000", prob2
        else:
            return "Diagnóstico no disponible", "No se pudo determinar el diagnóstico con la información proporcionada.", "❓", "#999999", 0.0
    except Exception as e:
        return "Diagnóstico no disponible", f"Error al procesar los resultados: {e}", "❗", "#999999", 0.0


def extraer_preguntas_relevantes(registro, etiquetas):
    """
    Devuelve hasta 5 preguntas consideradas más relevantes para un registro.

    Criterios:
    - Respuestas binarias con valor 1
    - Campos de texto no vacíos (y no puramente numéricos)
    """
    relevantes = []

    for campo, valor in registro.items():
        if campo in ["Registrado por", "ID"]:
            continue
        if pd.isna(valor):
            continue
        valor_str = str(valor).strip()
        if valor_str == "1":
            relevantes.append((campo, etiquetas.get(campo, campo)))
        elif not valor_str.isdigit() and len(valor_str) > 1:
            relevantes.append((campo, etiquetas.get(campo, campo)))

    return relevantes[:5]

def mostrar_resultado_prediccion(fila: dict, variables_importantes=None):
    diagnostico, mensaje, emoji, color, probabilidad = analizar_diagnostico(fila)

    # Mostrar el bloque visual
    st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px;
                    border-left: 5px solid {color}; margin-bottom:20px;'>
            <h3 style='color:{color}; margin-top:0;'>{emoji} Diagnóstico: {diagnostico}</h3>
            <p style='margin-bottom:0;'>{mensaje}</p>
        </div>
    """, unsafe_allow_html=True)

    texto_a_leer = mensaje

    # Variables importantes
    if variables_importantes:
        st.markdown("#### 🔍 Factores más relevantes en esta evaluación:")
        texto_a_leer += " Factores relevantes considerados fueron: "
        for var, val in variables_importantes:
            st.markdown(f"- **{var}**: {val}")
            texto_a_leer += f"{var}, "

    # Lectura en voz
    if st.session_state.get("voz_activa", False):
        leer_en_voz(texto_a_leer.strip())

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
    modelo = cargar_modelo1()
    proba = modelo.predict_proba(X)[0, 1]
    pred = int(proba >= 0.21)
    mostrar_resultado_prediccion(proba, pred)

def nuevo_registro():
    st.title("📝 Registro de Pacientes")

    if st.session_state.get("voz_activa", False):
        leer_en_voz("Estás en la sección de registro de pacientes. Por favor responde las siguientes preguntas.")

    # Cargar preguntas del formulario
    with open(RUTA_PREGUNTAS, encoding="utf-8") as f:
        secciones = json.load(f)

    respuestas = {}

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
                        codigo = p.get("codigo") or f"preg_{uuid.uuid4().hex[:6]}"
                        if st.session_state.get("voz_activa", False):
                            leer_en_voz(p.get("label", ""))
                        respuestas[codigo] = render_pregunta(p, key=codigo)
            else:
                for i, p in enumerate(preguntas):
                    codigo = p.get("codigo") or f"preg_{uuid.uuid4().hex[:6]}"
                    if st.session_state.get("voz_activa", False):
                        leer_en_voz(p.get("label", ""))
                    respuestas[codigo] = render_pregunta(p, key=codigo)

        if st.form_submit_button("Guardar"):
            # Validar y limpiar las respuestas del formulario
            respuestas_limpias = {}
            for k, v in respuestas.items():
                if v is None or v == "":
                    respuestas_limpias[k] = -1  # valor por defecto
                else:
                    try:
                        respuestas_limpias[k] = float(v) if k in COLUMNAS_MODELO else v
                    except ValueError:
                        respuestas_limpias[k] = -1  # valor por defecto si no puede convertirse

            df_modelo = pd.DataFrame([respuestas_limpias])
            resultado = predecir_nuevos_registros(df_modelo)

            fila_final = resultado.iloc[0].to_dict()
            pred1 = int(fila_final.get("Predicción Óptima 1", 0))

            if pred1 == 1:
                modelo = cargar_modelo2()
            else:
                modelo = cargar_modelo1()

            # Guardar en Sheets
            guardar_respuesta_paciente(fila_final)

            # Mostrar resultado
            st.success("✅ Registro guardado correctamente.")
            if st.session_state.get("voz_activa", False):
                leer_en_voz("Registro guardado correctamente. Mostrando resultados.")
            mostrar_resultado_prediccion(fila_final)
            st.rerun()

def mostrar_pacientes():
    import os

    def mostrar_recomendaciones_pdf(estado: str):
        temas = ["Ejercicio", "Habitos", "Nutricion"]

        estado_archivo = {
            "Sano": "Sanos",
            "Prediabético": "Prediabetes",
            "Diabético": "Diabetes"
        }.get(estado, "Sanos")

        st.markdown("### 📥 Recomendaciones personalizadas")
        for tema in temas:
            nombre_archivo = f"{tema} ({estado_archivo}).pdf"
            ruta = os.path.join(nombre_archivo)

            try:
                with open(ruta, "rb") as f:
                    st.download_button(
                        label=f"Descargar Recomendaciones de {tema}",
                        data=f,
                        file_name=nombre_archivo,
                        mime="application/pdf"
                    )
            except FileNotFoundError:
                st.warning(f"Archivo no disponible: {nombre_archivo}")

    st.title("📋 Participante")

    if st.session_state.get("voz_activa", False):
        leer_en_voz("Estás en la sección de participantes. Aquí puedes consultar los registros guardados.")

    sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
    df = pd.DataFrame(sheet.get_all_records())
    usuario = st.session_state.get("usuario", "").strip().lower()
    df = df[df["Registrado por"].str.strip().str.lower() == usuario]

    if df.empty:
        st.info("Todavía no tienes registros.")
        return

    df["ID"] = ["Registro #" + str(i + 1) for i in df.index]
    seleccionado = st.selectbox("Selecciona un registro:", ["Selecciona"] + df["ID"].tolist())

    if seleccionado == "Selecciona":
        return

    registro = df[df["ID"] == seleccionado].iloc[0]
    st.subheader(f"🧾 {seleccionado}")

    # Diagnóstico
    diagnostico, mensaje, emoji, color, _ = analizar_diagnostico(registro)
    estado = diagnostico.replace("Perfil ", "") if "Perfil" in diagnostico else diagnostico
    mostrar_relevantes = estado in ["Prediabético", "Diabético"]

    st.markdown(f"""
        <div style='background-color:#f8f9fa; padding:20px; border-radius:10px;
                    border-left: 6px solid {color}; margin-bottom:20px;'>
            <h4 style='color:{color}; margin-top:0;'>{emoji} {mensaje}</h4>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("voz_activa", False):
        leer_en_voz(mensaje)

    mostrar_recomendaciones_pdf(estado)

    etiquetas = {}
    valores_a_texto = {}

    try:
        with open(RUTA_PREGUNTAS, encoding="utf-8") as f:
            preguntas_json = json.load(f)

        def extraer_mapeo(data):
            if isinstance(data, dict):
                for v in data.values():
                    extraer_mapeo(v)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "codigo" in item:
                        codigo = item["codigo"]
                        etiquetas[codigo] = item.get("label", codigo)
                        if "valores" in item and "opciones" in item:
                            valores_a_texto[codigo] = {
                                str(val): texto for val, texto in zip(item["valores"], item["opciones"])
                            }

        extraer_mapeo(preguntas_json)

    except Exception as e:
        st.warning(f"No se pudo cargar etiquetas ni valores desde el JSON: {e}")

    def extraer_preguntas_relevantes(registro, etiquetas):
        relevantes = []
        for campo, valor in registro.items():
            if campo in ["Registrado por", "ID"]:
                continue
            if pd.isna(valor):
                continue
            valor_str = str(valor).strip()
            if valor_str == "1":
                relevantes.append((campo, etiquetas.get(campo, campo)))
            elif not valor_str.isdigit() and len(valor_str) > 1:
                relevantes.append((campo, etiquetas.get(campo, campo)))
        return relevantes[:5]

    if mostrar_relevantes:
        relevantes = extraer_preguntas_relevantes(registro, etiquetas)
        if relevantes:
            texto_html = "<ul style='margin-top: 0;'>"
            texto_relevante = "Preguntas más relevantes: "

            for campo, etiqueta in relevantes:
                valor_str = str(registro.get(campo, "")).strip()
                if campo in valores_a_texto:
                    texto_valor = valores_a_texto[campo].get(valor_str, valor_str)
                elif campo == "sexo":
                    texto_valor = "Hombre" if valor_str == "1" else "Mujer" if valor_str == "2" else valor_str
                else:
                    texto_valor = valor_str
                texto_html += f"<li><b>{etiqueta}</b>: {texto_valor}</li>"
                texto_relevante += f"{etiqueta}, "

            texto_html += "</ul>"

            st.markdown(f"""
                <div style='background-color:#f8f9fa; padding:20px; border-radius:10px;
                            border-left: 6px solid #007BFF; margin-bottom:20px;'>
                    <h4 style='color:#007BFF; margin-top:0;'>⭐ Preguntas más relevantes en este registro</h4>
                    {texto_html}
                </div>
            """, unsafe_allow_html=True)

            if st.session_state.get("voz_activa", False):
                leer_en_voz(texto_relevante.strip())

    st.markdown("### ✍🏽 Respuestas registradas")
    for campo, valor in registro.items():
        if campo in ["Registrado por", "ID"] or pd.isna(valor):
            continue

        label = etiquetas.get(campo, campo)
        valor_str = str(valor).strip()
        if campo in valores_a_texto:
            texto_valor = valores_a_texto[campo].get(valor_str, valor_str)
        elif campo == "sexo":
            texto_valor = "Hombre" if valor_str == "1" else "Mujer" if valor_str == "2" else valor_str
        elif campo.startswith("Predicción") or campo.startswith("Probabilidad"):
            continue
        else:
            texto_valor = valor_str

        st.markdown(f"**{label}:** {texto_valor}")

def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
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

