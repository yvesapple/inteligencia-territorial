import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Inteligencia Territorial Nocturna - CABA",
    page_icon="🌃",
    layout="wide",
)

# Constantes de la grilla oficial de CABA
LAT_MIN, LAT_MAX = -34.7050, -34.5265
LON_MIN, LON_MAX = -58.5315, -58.3350
TAM_LAT, TAM_LON = 0.0045, 0.0055

ZONAS_REFERENCIA = {
    "Palermo (Plaza Serrano / Soho)": (-34.5880, -58.4160),
    "Microcentro / Obelisco": (-34.6037, -58.3816),
    "Recoleta (Av. Santa Fe y Pueyrredón)": (-34.5945, -58.4020),
    "Belgrano (Barrancas / Cabildo)": (-34.5600, -58.4550),
    "San Telmo (Plaza Dorrego)": (-34.6212, -58.3731),
    "Caballito (Acoyte y Rivadavia)": (-34.6180, -58.4360),
    "Puerto Madero": (-34.6118, -58.3622),
    "Liniers (Centro de Trasbordo)": (-34.6390, -58.5260),
}


@st.cache_data
def cargar_datos_locales():
    rutas_posibles = [
        "data/processed/dataset_inteligencia_territorial.csv",
        "actor_apify/my_actor/dataset_inteligencia_territorial.csv",
        "dataset_inteligencia_territorial.csv",
    ]
    ruta_csv = next((r for r in rutas_posibles if os.path.exists(r)), None)
    if not ruta_csv:
        raise FileNotFoundError(f"No se encontró el dataset maestro en: {rutas_posibles}")

    df = pd.read_csv(ruta_csv, sep=None, engine="python")
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ["lat_min", "lat_max", "lon_min", "lon_max", "centro_lat", "centro_lon", "delitos_nocturnos"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
    col_lum = next((c for c in df.columns if "lum" in c), None)
    if col_lum:
        df["luminosidad_promedio"] = pd.to_numeric(
            df[col_lum].astype(str).str.replace(",", "."), errors="coerce"
        ).fillna(75.43)
    else:
        df["luminosidad_promedio"] = 75.43
    return df


@st.cache_data
def cargar_modelo_local():
    rutas_posibles = [
        "actor_apify/my_actor/modelo_riesgo_caba.json",
        "modelo_riesgo_caba.json",
    ]
    ruta_json = next((r for r in rutas_posibles if os.path.exists(r)), None)
    if ruta_json:
        with open(ruta_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def clasificar_luminosidad(valor: float) -> str:
    if valor < 60.3:
        return "Baja"
    elif valor <= 83.0:
        return "Media"
    return "Alta"


def predecir_local(df_zonas, artefacto, lat_raw, lon_raw, dia_semana, hora):
    gi = int(np.floor((lat_raw - LAT_MIN + 1e-6) / TAM_LAT))
    gj = int(np.floor((lon_raw - LON_MIN + 1e-6) / TAM_LON))
    z_lat = round(LAT_MIN + gi * TAM_LAT, 4)
    z_lon = round(LON_MIN + gj * TAM_LON, 4)
    zona_id = f"{z_lat:.4f}_{z_lon:.4f}"
    lat_c = round(z_lat + TAM_LAT / 2, 6)
    lon_c = round(z_lon + TAM_LON / 2, 6)

    cond = (
        (df_zonas["lat_min"] <= lat_c) & (lat_c <= df_zonas["lat_max"]) &
        (df_zonas["lon_min"] <= lon_c) & (lon_c <= df_zonas["lon_max"])
    )
    fila = df_zonas[cond]

    if not fila.empty:
        d = fila.iloc[0]
        lum = round(float(d["luminosidad_promedio"]), 2)
        delitos = int(d["delitos_nocturnos"])
        nivel_base = str(d.get("nivel_incidencia", "Bajo"))
        valida = True
    else:
        lum, delitos, nivel_base, valida = 75.43, 0, "Bajo", False

    es_fds = 1 if dia_semana.lower() in {"sábado", "sabado", "domingo"} else 0

    if artefacto and "arboles" in artefacto:
        vector_x = [lat_c, lon_c, lum, hora, es_fds]
        clases = artefacto["clases"]
        probas = np.zeros(len(clases), dtype=float)
        for arbol in artefacto["arboles"]:
            nodo = 0
            while arbol["children_left"][nodo] != -1:
                if vector_x[arbol["feature"][nodo]] <= arbol["threshold"][nodo]:
                    nodo = arbol["children_left"][nodo]
                else:
                    nodo = arbol["children_right"][nodo]
            probas += np.array(arbol["value"][nodo], dtype=float)
        probas /= len(artefacto["arboles"])
        idx = int(np.argmax(probas))
        nivel_est = str(clases[idx])
        confianza = round(float(probas[idx]), 4)
        modelo = artefacto.get("modelo_nombre", "RandomForest")
    else:
        nivel_est, confianza, modelo = nivel_base, 0.75, "HeuristicaFallback"

    return {
        "zona_id": zona_id,
        "latitud": lat_c,
        "longitud": lon_c,
        "zona_valida_en_dataset": valida,
        "dia_semana": dia_semana,
        "franja_horaria": hora,
        "nivel_incidencia_estimado": nivel_est,
        "confianza_modelo": confianza,
        "modelo_utilizado": modelo,
        "luminosidad_promedio": lum,
        "clasificacion_luminosidad": clasificar_luminosidad(lum),
        "delitos_nocturnos_historicos": delitos,
    }


def predecir_apify(token, actor_id, zona_id, dia_semana, hora):
    actor_slug = actor_id.replace("/", "~")
    url = f"https://api.apify.com/v2/acts/{actor_slug}/run-sync-get-dataset-items?token={token}"
    payload = {
        "operation": "inference",
        "bulk_items": [
            {"zona_id": zona_id, "dia_semana": dia_semana, "franja_horaria": hora}
        ],
    }
    resp = requests.post(url, json=payload, timeout=90)
    resp.raise_for_status()
    items = resp.json()
    if not items:
        raise ValueError("Apify devolvió una respuesta vacía.")
    return items[0]


# --- INICIALIZACIÓN DE ESTADO PARA EL MAPA Y COORDENADAS ---
if "lat_sel" not in st.session_state:
    st.session_state["lat_sel"] = -34.5880
if "lon_sel" not in st.session_state:
    st.session_state["lon_sel"] = -58.4160
if "ultimo_clic" not in st.session_state:
    st.session_state["ultimo_clic"] = None


def actualizar_desde_referencia():
    lat_r, lon_r = ZONAS_REFERENCIA[st.session_state["zona_ref_box"]]
    st.session_state["lat_sel"] = lat_r
    st.session_state["lon_sel"] = lon_r
    st.session_state["ultimo_resultado"] = None


# --- INTERFAZ PRINCIPAL ---
df_zonas = cargar_datos_locales()
artefacto = cargar_modelo_local()

st.title("🌃 Plataforma de Inteligencia Territorial Nocturna — CABA")
st.markdown("Evaluación de riesgo nocturno combinando **datos abiertos de delitos (GCBA)** e **imágenes satelitales NASA Black Marble**.")

with st.sidebar:
    st.header("⚙️ Motor de Inferencia")
    modo = st.radio(
        "Modo de ejecución:",
        ["⚡ Inferencia Local (Baja Latencia)", "☁️ API Apify en Vivo (Cloud)"],
    )
    apify_token = ""
    apify_actor_id = ""
    if "Apify" in modo:
        apify_token = st.text_input("APIFY_TOKEN", type="password", placeholder="apify_api_...")
        apify_actor_id = st.text_input("Actor ID (usuario~nombre)", placeholder="tu-usuario~nombre-actor")

    st.divider()
    st.header("📍 Parámetros de Consulta")
    st.selectbox(
        "Zona de referencia rápida:",
        list(ZONAS_REFERENCIA.keys()),
        key="zona_ref_box",
        on_change=actualizar_desde_referencia,
    )

    lat_in = st.number_input("Latitud", key="lat_sel", format="%.4f", step=0.0010)
    lon_in = st.number_input("Longitud", key="lon_sel", format="%.4f", step=0.0010)
    dia_in = st.selectbox("Día de la semana", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"], index=5)
    hora_in = st.slider("Franja horaria (0 a 23 hs)", min_value=0, max_value=23, value=23)

tab_pred, tab_eda = st.tabs(["🔮 Producto 2: Predicción de Riesgo (Inference)", "📊 Producto 1: Exploración Territorial (Analytics)"])

with tab_pred:
    col_mapa, col_res = st.columns([1.3, 1])

    with col_mapa:
        st.subheader("Mapa Interactivo de CABA")
        st.caption("Hacé clic sobre cualquier punto del mapa para mover el marcador y evaluar esa celda.")
        m = folium.Map(location=[lat_in, lon_in], zoom_start=13, tiles="OpenStreetMap")

        top_celdas = df_zonas[df_zonas["delitos_nocturnos"] > 0].nlargest(120, "delitos_nocturnos")
        for _, r in top_celdas.iterrows():
            color = "#d9534f" if str(r.get("nivel_incidencia")) == "Alto" else "#f0ad4e"
            folium.Rectangle(
                bounds=[[r["lat_min"], r["lon_min"]], [r["lat_max"], r["lon_max"]]],
                color=color,
                weight=1,
                fill=True,
                fill_opacity=0.25,
                tooltip=f"Zona {r.get('zona_id', '')} | Delitos nocturnos: {int(r['delitos_nocturnos'])}",
            ).add_to(m)

        folium.Marker(
            [lat_in, lon_in],
            popup=f"Ubicación seleccionada ({lat_in:.4f}, {lon_in:.4f})",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(m)

        map_data = st_folium(m, height=440, width="100%", key="mapa_caba")

        if map_data and map_data.get("last_clicked"):
            clic_actual = (
                round(map_data["last_clicked"]["lat"], 4),
                round(map_data["last_clicked"]["lng"], 4),
            )
            if clic_actual != st.session_state["ultimo_clic"]:
                st.session_state["ultimo_clic"] = clic_actual
                st.session_state["lat_sel"] = clic_actual[0]
                st.session_state["lon_sel"] = clic_actual[1]
                st.session_state["ultimo_resultado"] = None
                st.rerun()

    with col_res:
        st.subheader("Resultado del Modelo")
        zona_query = f"{lat_in:.4f}_{lon_in:.4f}"

        if st.button("🚀 Evaluar Riesgo de la Zona", type="primary", use_container_width=True):
            try:
                if "Apify" in modo:
                    if not apify_token or not apify_actor_id:
                        st.error("Completá tu APIFY_TOKEN y Actor ID en el panel izquierdo.")
                        st.stop()
                    with st.spinner("Consultando Actor en Apify..."):
                        res = predecir_apify(apify_token, apify_actor_id, zona_query, dia_in, hora_in)
                else:
                    res = predecir_local(df_zonas, artefacto, lat_in, lon_in, dia_in, hora_in)

                st.session_state["ultimo_resultado"] = res
            except Exception as e:
                st.error(f"Error al consultar el modelo: {e}")

        res = st.session_state.get("ultimo_resultado") or predecir_local(df_zonas, artefacto, lat_in, lon_in, dia_in, hora_in)

        nivel = res["nivel_incidencia_estimado"]
        color_badge = {"Alto": "🔴", "Medio": "🟠", "Bajo": "🟢"}.get(nivel, "⚪")

        st.metric("Nivel de Riesgo Estimado", f"{color_badge} {nivel.upper()}", f"Confianza: {res['confianza_modelo']*100:.1f}%")

        c1, c2 = st.columns(2)
        c1.metric("Luminosidad Satelital", f"{res['luminosidad_promedio']} nW", res["clasificacion_luminosidad"])
        c2.metric("Delitos Nocturnos (2024)", f"{res['delitos_nocturnos_historicos']}", f"Celda {res['zona_id']}")

        st.caption(f"Motor: `{res['modelo_utilizado']}` | Zona válida en grilla: `{res['zona_valida_en_dataset']}`")

        with st.expander("Ver respuesta JSON cruda (formato API Apify)"):
            st.json(res)

with tab_eda:
    st.subheader("Resumen Ejecutivo del Territorio (1.600 celdas)")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Celdas Analizadas", f"{len(df_zonas):,}")
    k2.metric("Total Delitos Nocturnos", f"{int(df_zonas['delitos_nocturnos'].sum()):,}")
    k3.metric("Luminosidad Media CABA", f"{df_zonas['luminosidad_promedio'].mean():.2f}")

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Distribución por Nivel de Incidencia**")
        if "nivel_incidencia" in df_zonas.columns:
            resumen_inc = (
                df_zonas.groupby("nivel_incidencia", as_index=False)
                .agg(celdas=("delitos_nocturnos", "count"), delitos_totales=("delitos_nocturnos", "sum"))
            )
            st.dataframe(resumen_inc, use_container_width=True, hide_index=True)

    with c_right:
        st.markdown("**Top 10 Celdas con Mayor Incidencia Nocturna**")
        cols_top = [c for c in ["zona_id", "delitos_nocturnos", "luminosidad_promedio", "nivel_incidencia"] if c in df_zonas.columns]
        st.dataframe(df_zonas.sort_values("delitos_nocturnos", ascending=False).head(10)[cols_top], use_container_width=True, hide_index=True)
