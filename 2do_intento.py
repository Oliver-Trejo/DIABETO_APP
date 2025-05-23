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
import numpy as np

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

def obtener_variables_importantes(modelo, datos):
    # Extrae el último paso del pipeline que tenga feature_importances_
    modelo_final = None
    if hasattr(modelo, "named_steps"):
        for step in reversed(modelo.named_steps.values()):
            if hasattr(step, "feature_importances_"):
                modelo_final = step
                break
    elif hasattr(modelo, "feature_importances_"):
        modelo_final = modelo

    if modelo_final is None:
        st.warning("⚠️ El modelo no tiene feature_importances_.")
        return []

    importancias = modelo_final.feature_importances_
    top_indices = importancias.argsort()[::-1]

    fila = datos.iloc[0].to_dict()
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


def mostrar_resultado_prediccion(fila: dict, variables_importantes=None):
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # Determinar diagnóstico
    try:
        pred1 = int(fila.get("Predicción Óptima 1", 0))
        prob1 = safe_float(fila.get("Probabilidad Estimada 1"))
        pred2 = fila.get("Predicción Óptima 2", "")
        prob2 = safe_float(fila.get("Probabilidad Estimada 2"))

        if pred1 == 0:
            diagnostico = "Sano"
            probabilidad = prob1
            color = "#4CAF50"
            emoji = "✅"
            mensaje = "¡Buenas noticias! No encontramos señales claras de diabetes. Aun así, cuida tu salud."
        elif str(pred2) == "0":
            diagnostico = "Prediabético"
            probabilidad = prob2
            color = "#FFA500"
            emoji = "🟠"
            mensaje = "Tus respuestas indican señales compatibles con una condición prediabética. Te recomendamos consultar a un especialista."
        elif str(pred2) == "1":
            diagnostico = "Diabético"
            probabilidad = prob2
            color = "#FF0000"
            emoji = "🚨"
            mensaje = "Tus respuestas indican señales compatibles con diabetes tipo 2. Es importante que acudas a un centro de salud lo antes posible."
        else:
            diagnostico = "Diagnóstico no disponible"
            probabilidad = 0
            color = "#999999"
            emoji = "❓"
            mensaje = "No se pudo determinar el diagnóstico con la información proporcionada."

    except Exception as e:
        diagnostico = "Diagnóstico no disponible"
        probabilidad = 0
        color = "#999999"
        emoji = "❗"
        mensaje = f"Error al procesar los resultados: {e}"

    # Mostrar el bloque visual
    st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px;
                    border-left: 5px solid {color}; margin-bottom:20px;'>
            <h3 style='color:{color}; margin-top:0;'>{emoji} Diagnóstico: {diagnostico}</h3>
            <p style='margin-bottom:0;'>{mensaje}</p>
            <p style='font-weight:bold;'>Probabilidad estimada: {probabilidad:.2%}</p>
        </div>
    """, unsafe_allow_html=True)

    texto_a_leer = f"{mensaje} Tu probabilidad estimada es del {probabilidad:.0%}. "

    # Variables importantes
    if variables_importantes:
        st.markdown("#### 🔍 Factores más relevantes en esta evaluación:")
        texto_a_leer += "Factores relevantes considerados fueron: "
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

            variables_relevantes = obtener_variables_importantes(modelo, resultado)

            # Guardar en Sheets
            guardar_respuesta_paciente(fila_final)

            # Mostrar resultado
            st.success("✅ Registro guardado correctamente.")
            if st.session_state.get("voz_activa", False):
                leer_en_voz("Registro guardado correctamente. Mostrando resultados.")
            mostrar_resultado_prediccion(fila_final, variables_relevantes)
            st.rerun()

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

        # Mapeo etiquetas y valores
        etiquetas = {}
        valores_a_texto = {}

        for seccion in preguntas.values():
            if isinstance(seccion, list):
                for p in seccion:
                    if "codigo" in p:
                        etiquetas[p["codigo"]] = p.get("label", p["codigo"])
                        if "valores" in p and "opciones" in p:
                            valores_a_texto[p["codigo"]] = dict(zip(
                                map(str, p["valores"]),
                                p["opciones"]
                            ))
            elif isinstance(seccion, dict):
                for grupo in seccion.values():
                    for p in grupo:
                        if "codigo" in p:
                            etiquetas[p["codigo"]] = p.get("label", p["codigo"])
                            if "valores" in p and "opciones" in p:
                                valores_a_texto[p["codigo"]] = dict(zip(
                                    map(str, p["valores"]),
                                    p["opciones"]
                                ))

        # Determinar diagnóstico
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

        # Mostrar diagnóstico y factores importantes
        st.markdown("### 📊 Resultado de Evaluación")
        try:
            df_modelo = pd.DataFrame([{k: v for k, v in registro.items() if k in COLUMNAS_MODELO}])

            if df_modelo["sexo"].iloc[0] not in [1, 2]:
                df_modelo["sexo"] = df_modelo["sexo"].replace({"Hombre": 1, "Mujer": 2}).fillna(-1)

            df_modelo = df_modelo.replace("", np.nan)
            X = df_modelo[COLUMNAS_MODELO].apply(pd.to_numeric, errors="coerce").fillna(-1)
            variables_relevantes = obtener_variables_importantes(modelo, X)

            # ✅ CORREGIDO: Argumentos nombrados
            mostrar_resultado_prediccion(
                pred=pred,
                modelo_usado=modelo_usado,
                variables_importantes=variables_relevantes
            )
        except Exception as e:
            st.error(f"Error al generar diagnóstico: {e}")
            if st.session_state.get("voz_activa", False):
                leer_en_voz("Ocurrió un error al procesar el resultado del participante.")

        # Respuestas registradas
        st.markdown("### ✍🏽 Respuestas Registradas")
        for campo, valor in registro.items():
            if campo in ["Registrado por", "ID"] or pd.isna(valor):
                continue

            label = etiquetas.get(campo, campo.replace("_", " ").title())
            texto_valor = str(valor)

            # Traducir si existe mapeo
            if campo in valores_a_texto:
                texto_valor = valores_a_texto[campo].get(str(valor), texto_valor)

            elif campo == "sexo":
                texto_valor = "Hombre" if str(valor) in ["1", "Hombre"] else "Mujer" if str(valor) in ["2", "Mujer"] else texto_valor

            elif campo.startswith("Predicción") or campo.startswith("Probabilidad"):
                continue  # Ya mostrado arriba

            st.markdown(f"**{label}:** {texto_valor}")

        # Descargar PDF
        st.download_button(
            label="📥 Descargar informe completo",
            data=generar_pdf(
                [(etiquetas.get(k, k), valores_a_texto.get(k, {}).get(str(v), v))
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
