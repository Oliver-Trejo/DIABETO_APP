# ✅ ARCHIVO: interfaz.py (versión optimizada)
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
import folium
from streamlit_folium import folium_static

# --- CONFIGURACIONES GLOBALES ---
st.set_page_config(page_title="DIABETO", page_icon="🏥", layout="wide")
RUTA_PREGUNTAS = "preguntas_con_codigos.json"
RUTA_CREDENCIALES = "credentials.json"
COLUMNAS_MODELO = [
    "SEXO", "PESO1_1", "TALLA4_1", "P27_1_1", "P27_1_2", "P1_1", "P1_6", "P4_1", "P1_7",
    "P5_1", "P5_2_1", "P6_1_1", "P6_6", "P6_4", "P13_1", "P13_2", "P13_10", "P13_11", "P13_12_1",
    "P7_1_1", "P7_2_1", "P7_3_1", "P7_5_1", "P7_1_2", "P7_2_2", "P7_3_2", "P7_5_2",
    "P7_1_3", "P7_2_3", "P7_3_3", "P7_5_3", "COMIDA_RAP", "DULCES", "CEREALES_DUL"
]

# Importación de modelo
RUTA_MODELO = "modelo_rf_entrenado.pkl"

# API PLACE MAPA

@st.cache_resource
def cargar_modelo():
    return joblib.load(RUTA_MODELO)

# --- FUNCIONES AUXILIARES ---

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
    importancias = modelo.feature_importances_
    top_indices = importancias.argsort()[::-1]

    fila = datos.iloc[0].to_dict()  # Convertir a diccionario

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

    modo = st.radio("Selecciona una opción:", ["Iniciar sesión", "Crear cuenta"])

    if modo == "Iniciar sesión":
        with st.form("login_form"):
            nombre = st.text_input("Nombre completo", key="login_nombre")
            password = st.text_input("Contraseña", type="password", key="login_pass")
            if st.form_submit_button("Ingresar"):
                usuario = buscar_usuario_por_nombre(nombre)
                if usuario and usuario["Contraseña Hasheada"] == hash_password(password):
                    st.session_state["logged_in"] = True
                    st.session_state["usuario"] = nombre  # ✅ ESTA ES LA CLAVE
                    st.sidebar.markdown(f"👤 Sesión activa: **{st.session_state['usuario']}**")
                    st.success(f"Bienvenido, {nombre}")
                    st.rerun()
                else:
                    st.error("No pudimos encontrar tus datos. Revisa que estén bien escritos o intenta registrarte.")

    elif modo == "Crear cuenta":
        with st.form("registro_form"):
            nombre = st.text_input("Nombre completo", key="reg_nombre")
            password = st.text_input("Contraseña", type="password", key="reg_pass")
            if st.form_submit_button("Registrar"):
                if buscar_usuario_por_nombre(nombre):
                    st.error("Este nombre ya fue usado. Prueba con uno diferente.")
                elif not nombre or not password:
                    st.warning("Te falta llenar algún dato. Revisa por favor.")
                else:
                    registrar_usuario(nombre, password)
                    st.success("Cuenta creada correctamente. Ya puedes iniciar sesión.")
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

    nombre_usuario = st.session_state.get("usuario", "Usuario")
    st.markdown(f"<h2>{nombre_usuario}</h2>", unsafe_allow_html=True)

    try:
        with open("intro_text.json", encoding="utf-8") as f:
            textos = json.load(f)
            texto_html = f"<div class='texto-introductorio'>{textos.get('mi_cuenta', '')}</div>"
            st.markdown(texto_html, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("No se encontró el archivo de texto introductorio.")

    st.markdown("</div>", unsafe_allow_html=True)

def mostrar_pacientes():
    st.title("📋 Participante")

    # ✅ Cargar pacientes
    sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
    df = pd.DataFrame(sheet.get_all_records())

    usuario = st.session_state.get("usuario", "").strip().lower()
    df = df[df["Registrado por"].str.strip().str.lower() == usuario]

    if df.empty:
        st.info("Todavía no hay ningún registro guardado. Puedes crear uno en la sección de ‘Nuevo Registro’")
        return

    df = df.dropna(how="all").reset_index(drop=True)
    df["ID Paciente"] = ["Registro #" + str(i + 1) for i in df.index]
    seleccionado = st.selectbox("Selecciona un registro para ver el detalle:", ["Selecciona"] + df["ID Paciente"].tolist())

    if seleccionado != "Selecciona":
        idx = df[df["ID Paciente"] == seleccionado].index[0]
        registro = df.iloc[idx]

        st.subheader(f"🧾 {seleccionado}")

        with open(RUTA_PREGUNTAS, encoding="utf-8") as f:
            preguntas_json = json.load(f)

        codigo_a_label = {}
        codigo_a_opciones = {}

        for bloque in ["Generales", "Salud", "Hábitos Alimenticios"]:
            for p in preguntas_json.get(bloque, []):
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

        # Diagnóstico
        variables_etiquetadas = []
        if "Probabilidad Estimada" in registro and "Predicción Óptima" in registro:
            prob = float(registro["Probabilidad Estimada"])
            pred = int(registro["Predicción Óptima"])
            modelo = cargar_modelo()
            df_modelo = registro.to_frame().T
            df_modelo["SEXO"] = df_modelo["SEXO"].replace({"Masculino": 1, "Femenino": 0, "Otro": 2})
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

            mostrar_resultado_prediccion(prob, pred, variables_etiquetadas)

        # 🌍 Mostrar ubicación con botón
        st.markdown("#### 🌍 ¿Quieres ver tu ubicación en el mapa y encontrar los Centros de Salud más cercanos?🏥")
        location = streamlit_geolocation()
        if location and location.get("latitude") and location.get("longitude"):
            lat = location["latitude"]
            lon = location["longitude"]
            #st.success(f"✅ Coordenadas obtenidas:\nLatitud: {lat}\nLongitud: {lon}")

            # Mostrar mapa con marcador
            mapa = folium.Map(location=[lat, lon], zoom_start=16)
            folium.Marker([lat, lon], tooltip="📍 Aquí estás").add_to(mapa)
            folium_static(mapa)
        else:
            st.warning("⚠ Haz clic en el botón para ver en el mapa dónde estás. Así podremos ayudarte mejor.")

        # Mostrar respuestas completas
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

                # Botón para generar PDF de respuestas
        if st.button("📥 Descargar resumen de respuestas"):
            pdf_buffer = generar_pdf(respuestas_mostradas, variables_etiquetadas)
            st.download_button("Descargar respuestas en PDF", data=pdf_buffer, file_name=f"{seleccionado}.pdf", mime="application/pdf")

        # PDF de recomendaciones personalizadas
        st.markdown("#### 📄 Recomendaciones para ti")

        if pred == 1:
            col1, col2, col3 = st.columns(3)
            with col1:
                with open("nutricion_diabeticos.txt", "rb") as f:
                    st.download_button("🍽️ Nutrición", f, file_name="nutricion_diabeticos.txt")
            with col2:
                with open("ejercicio_diabeticos.txt", "rb") as f:
                    st.download_button("🏃‍♀️ Ejercicio", f, file_name="ejercicio_diabeticos.txt")
            with col3:
                with open("habitos_diabeticos.txt", "rb") as f:
                    st.download_button("🌱 Hábitos", f, file_name="habitos_diabeticos.txt")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                with open("nutricion_sanos.txt", "rb") as f:
                    st.download_button("🍽️ Nutrición", f, file_name="nutricion_sanos.txt")
            with col2:
                with open("ejercicio_sanos.txt", "rb") as f:
                    st.download_button("🏃‍♀️ Ejercicio", f, file_name="ejercicio_sanos.txt")
            with col3:
                with open("habitos_sanos.txt", "rb") as f:
                    st.download_button("🌱 Hábitos", f, file_name="habitos_sanos.txt")

                # Botón para eliminar registro
                if st.button("🗑️ Eliminar este registro"):
                    fila_real = idx + 2
                    sheet.delete_rows(fila_real)
                    st.success(f"✅ {seleccionado} eliminado.")
                    st.rerun()

def predecir_nuevos_registros(df_input, threshold=0.18):
    modelo = cargar_modelo()
    X = df_input[COLUMNAS_MODELO].replace("", -1).astype(float)
    df_input['Probabilidad Estimada'] = modelo.predict_proba(X)[:, 1]
    df_input['Predicción Óptima'] = (df_input['Probabilidad Estimada'] >= threshold).astype(int)
    return df_input

def guardar_respuesta_paciente(fila_dict, proba=None, pred=None):
    sheet = conectar_google_sheet(key=st.secrets["google_sheets"]["pacientes_key"])
    encabezados = sheet.row_values(1)

    # Asegurar que existan las claves requeridas
    fila_dict["Probabilidad Estimada"] = float(proba)
    fila_dict["Predicción Óptima"] = int(pred)
    fila_dict["Registrado por"] = st.session_state.get("usuario", "Desconocido")

    # Asegurar que latitud y longitud estén en los encabezados
    columnas_necesarias = ["latitud", "longitud"]
    for col in columnas_necesarias:
        if col not in encabezados:
            sheet.update_cell(1, len(encabezados) + 1, col)  # Añade al final
            encabezados.append(col)

    # Crear la nueva fila respetando el orden de encabezados
    nueva_fila = [fila_dict.get(col, "") for col in encabezados]
    sheet.append_row(nueva_fila)

def mostrar_resultado_prediccion(proba, pred, variables_importantes=None):
    color = "#FFA500" if pred == 1 else "#4CAF50"
    emoji = "⚠️" if pred == 1 else "✅"
    titulo = "Es importante que visites un centro de salud. Tus respuestas se parecen a las de personas con diabetes tipo 2." if pred == 1 else "¡Buenas noticias! No encontramos señales claras de diabetes. Aun así, cuida tu salud."
    st.markdown(f"""
        <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; border-left: 5px solid {color};'>
            <h3 style='color:{color};'>{emoji} {titulo}</h3>
            <p style='font-weight:bold;'>Tu perfil coincide con personas que tienen diabetes en un: {proba:.2%}</p>
        </div>
    """, unsafe_allow_html=True)
    if pred == 1 and variables_importantes:
        st.markdown("#### 🔍 Las siguientes respuestas fueron importantes para este resultado:")
        for var, val in variables_importantes:
            st.markdown(f"- **{var}**: {val}")

def ejecutar_prediccion():
    sheet = conectar_google_sheet(key="1C5H_AJQtMCvNdHfs55Hv8vl_LcwAI0_syK85JV1KUv0")
    df = pd.DataFrame(sheet.get_all_records())
    if df.empty:
        st.warning("No hay datos suficientes para predecir.")
        return
    faltantes = [col for col in COLUMNAS_MODELO if col not in df.columns]
    if faltantes:
        st.error(f"Faltan columnas: {faltantes}")
        return
    X = df.iloc[[-1]][COLUMNAS_MODELO].replace("", -1)
    modelo = cargar_modelo()
    proba = modelo.predict_proba(X)[0, 1]
    pred = int(proba >= 0.21)
    mostrar_resultado_prediccion(proba, pred)

def nuevo_registro():
    st.title("📝 Registro de Pacientes")

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
            if titulo == "Antecedentes familiares":
                for familiar, grupo in preguntas.items():
                    st.markdown(f"### {familiar}")
                    for i, p in enumerate(grupo):
                        codigo = p.get("codigo", f"{p['label']}-{i}")
                        respuestas[codigo] = render_pregunta(p, key=codigo)
            else:
                for i, p in enumerate(preguntas):
                    codigo = p.get("codigo", f"{p['label']}-{i}")
                    respuestas[codigo] = render_pregunta(p, key=codigo)

        if st.form_submit_button("Guardar"):
            # 🔹 Agregar ubicación detectada al registro
            respuestas["latitud"] = st.session_state.get("latitud", "")
            respuestas["longitud"] = st.session_state.get("longitud", "")

            # 🔹 Hacer predicción
            df_modelo = pd.DataFrame([respuestas])
            resultado = predecir_nuevos_registros(df_modelo)
            proba = resultado["Probabilidad Estimada"].iloc[0]
            pred = resultado["Predicción Óptima"].iloc[0]

            # 🔹 Guardar paciente
            guardar_respuesta_paciente(respuestas, proba, pred)

            st.success("✅ Registro guardado correctamente.")
            st.session_state["mostrar_prediccion"] = True
            modelo = cargar_modelo()
            variables_relevantes = obtener_variables_importantes(modelo, df_modelo)
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
