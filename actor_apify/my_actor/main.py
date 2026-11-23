import os
import json
import traceback
import numpy as np
import pandas as pd
import joblib
import base64
import io
from apify import Actor

# Parámetros oficiales del Notebook V2
ANIO_ANALISIS = 2024
LAT_MIN = -34.7050
LAT_MAX = -34.5265
LON_MIN = -58.5315
LON_MAX = -58.3350
HORA_NOCHE_INICIO = 20
HORA_NOCHE_FIN = 6
TAM_LAT = 0.0045
TAM_LON = 0.0055

COLUMNAS_ESPERADAS_RAW = [
    "id-mapa", "anio", "mes", "dia", "fecha", "franja",
    "tipo", "subtipo", "uso_arma", "uso_moto", "barrio",
    "comuna", "latitud", "longitud", "cantidad"
]


def clasificar_luminosidad(valor: float) -> str:
    if valor < 60.3:
        return "Baja"
    elif valor <= 83.0:
        return "Media"
    else:
        return "Alta"


def predecir_riesgo_temporal(nivel_base_zona: str, franja_horaria: int) -> str:
    if nivel_base_zona == "Sin registros nocturnos":
        return "Bajo"
    if franja_horaria >= HORA_NOCHE_INICIO or franja_horaria <= HORA_NOCHE_FIN:
        return nivel_base_zona
    if nivel_base_zona == "Alto":
        return "Medio"
    return "Bajo"

def ejecutar_etl(raw_records, df_maestro: pd.DataFrame) -> list:
    if isinstance(raw_records, str):
        raw_records = json.loads(raw_records)

    if not raw_records:
        raise ValueError("El parámetro 'raw_records' está vacío. Revisá el mapeo en Make.")

    df = pd.DataFrame(raw_records)
    df.columns = [str(c).strip().lower() for c in df.columns]
    Actor.log.info(f"Filas recibidas en ETL: {len(df)} | Columnas: {df.columns.tolist()}")

    # Si Make envió col1, col2... en lugar de los nombres reales, los renombramos por posición
    if "fecha" not in df.columns and "col5" in df.columns:
        mapeo_cols = {
            f"col{i + 1}": nombre
            for i, nombre in enumerate(COLUMNAS_ESPERADAS_RAW)
            if f"col{i + 1}" in df.columns
        }
        df = df.rename(columns=mapeo_cols)

    # Eliminar fila de encabezados si quedó incluida como dato
    if "fecha" in df.columns:
        df = df[df["fecha"].astype(str).str.lower() != "fecha"].copy()

    faltantes = [c for c in ["fecha", "franja", "latitud", "longitud"] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}. Columnas recibidas: {df.columns.tolist()}")

    # Limpiar comas decimales y espacios antes de convertir a número
    for col in ["anio", "franja", "comuna", "latitud", "longitud", "cantidad"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip().str.replace(",", "."), errors="coerce")

    # Parseo flexible de fechas (soporta tanto YYYY-MM-DD como DD/MM/YYYY)
    fechas_iso = pd.to_datetime(df["fecha"], format="%Y-%m-%d", errors="coerce")
    fechas_latam = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")
    df["fecha"] = fechas_iso.fillna(fechas_latam)

    # Filtro base consistente
    criterio_base = (
        df["fecha"].notna()
        & df["fecha"].dt.year.eq(ANIO_ANALISIS)
        & df["franja"].between(0, 23)
        & df["latitud"].between(LAT_MIN, LAT_MAX)
        & df["longitud"].between(LON_MIN, LON_MAX)
    )
    df_base = df.loc[criterio_base].copy()
    Actor.log.info(f"Filas válidas tras filtro base (2024 + CABA): {len(df_base)}")

    df_base["mes_num"] = df_base["fecha"].dt.month
    df_base["dia_semana_num"] = df_base["fecha"].dt.dayofweek
    df_base["es_fin_de_semana"] = df_base["dia_semana_num"].isin([5, 6]).astype(int)

    # Filtro período nocturno 20:00 a 06:59
    es_nocturno = (df_base["franja"] >= HORA_NOCHE_INICIO) | (df_base["franja"] <= HORA_NOCHE_FIN)
    df_nocturno = df_base.loc[es_nocturno].copy()
    Actor.log.info(f"Filas nocturnas (20:00-06:59) para agrupar: {len(df_nocturno)}")

    # Construcción de grilla territorial completa
    lat_edges = np.arange(LAT_MIN, LAT_MAX + TAM_LAT, TAM_LAT)
    lon_edges = np.arange(LON_MIN, LON_MAX + TAM_LON, TAM_LON)

    celdas = []
    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            lat0, lat1 = lat_edges[i], min(lat_edges[i + 1], LAT_MAX)
            lon0, lon1 = lon_edges[j], min(lon_edges[j + 1], LON_MAX)
            if lat0 >= LAT_MAX or lon0 >= LON_MAX:
                continue
            celdas.append({
                "zona_id": f"{lat0:.4f}_{lon0:.4f}",
                "lat_min": round(float(lat0), 4),
                "lat_max": round(float(lat1), 4),
                "lon_min": round(float(lon0), 4),
                "lon_max": round(float(lon1), 4),
                "centro_lat": round(float((lat0 + lat1) / 2), 6),
                "centro_lon": round(float((lon0 + lon1) / 2), 6),
            })

    df_grid = pd.DataFrame(celdas)

    if not df_nocturno.empty:
        df_nocturno["grid_i"] = np.floor((df_nocturno["latitud"] - LAT_MIN) / TAM_LAT).astype(int).clip(0, len(lat_edges) - 2)
        df_nocturno["grid_j"] = np.floor((df_nocturno["longitud"] - LON_MIN) / TAM_LON).astype(int).clip(0, len(lon_edges) - 2)
        df_nocturno["lat_min_c"] = LAT_MIN + df_nocturno["grid_i"] * TAM_LAT
        df_nocturno["lon_min_c"] = LON_MIN + df_nocturno["grid_j"] * TAM_LON
        df_nocturno["zona_id"] = (
            df_nocturno["lat_min_c"].round(4).map(lambda x: f"{x:.4f}")
            + "_"
            + df_nocturno["lon_min_c"].round(4).map(lambda x: f"{x:.4f}")
        )

        delitos_por_zona = df_nocturno.groupby("zona_id").size().reset_index(name="delitos_nocturnos")
        prop_fds = df_nocturno.groupby("zona_id")["es_fin_de_semana"].mean().round(4).reset_index(name="proporcion_fin_semana")
        hora_moda = (
            df_nocturno.groupby("zona_id")["franja"]
            .agg(lambda s: int(s.mode().iloc[0]) if not s.mode().empty else -1)
            .reset_index(name="hora_predominante")
        )

        df_grid = (
            df_grid.merge(delitos_por_zona, on="zona_id", how="left")
            .merge(prop_fds, on="zona_id", how="left")
            .merge(hora_moda, on="zona_id", how="left")
        )

        if "tipo" in df_nocturno.columns:
            tipo_pred = (
                df_nocturno.groupby("zona_id")["tipo"]
                .agg(lambda s: str(s.mode().iloc[0]) if not s.mode().empty else "Sin datos")
                .reset_index(name="tipo_predominante")
            )
            df_grid = df_grid.merge(tipo_pred, on="zona_id", how="left")
    else:
        df_grid["delitos_nocturnos"] = 0
        df_grid["proporcion_fin_semana"] = 0.0
        df_grid["hora_predominante"] = -1
        df_grid["tipo_predominante"] = "Sin datos"

    df_grid["delitos_nocturnos"] = df_grid["delitos_nocturnos"].fillna(0).astype(int)
    df_grid["proporcion_fin_semana"] = df_grid["proporcion_fin_semana"].fillna(0.0)
    df_grid["hora_predominante"] = df_grid["hora_predominante"].fillna(-1).astype(int)
    if "tipo_predominante" not in df_grid.columns:
        df_grid["tipo_predominante"] = "Sin datos"
    df_grid["tipo_predominante"] = df_grid["tipo_predominante"].fillna("Sin datos")

# Enriquecer con luminosidad satelital NASA Black Marble del dataset maestro
    col_lum = next((c for c in df_maestro.columns if "lum" in c or "rad" in c), None) if not df_maestro.empty else None

    if col_lum is not None:
        df_m = df_maestro.copy()
        df_m["lum_num"] = pd.to_numeric(
            df_m[col_lum].astype(str).str.strip().str.replace(",", "."), errors="coerce"
        )
        df_m = df_m.dropna(subset=["lum_num"]).copy()

        def norm_zid(z):
            try:
                lat_s, lon_s = str(z).strip().split("_")
                return f"{float(lat_s):.4f}_{float(lon_s):.4f}"
            except Exception:
                return str(z).strip()

        if "zona_id_norm" in df_m.columns:
            df_m["zona_id_4d"] = df_m["zona_id_norm"].apply(norm_zid)
            mapa_lum = df_m.groupby("zona_id_4d")["lum_num"].mean().to_dict()
            df_grid["luminosidad_promedio"] = df_grid["zona_id"].map(mapa_lum)
        else:
            df_grid["luminosidad_promedio"] = np.nan

        faltan_lum = df_grid["luminosidad_promedio"].isna()
        col_lat = next((c for c in df_m.columns if "lat" in c and "min" not in c and "max" not in c), None)
        col_lon = next((c for c in df_m.columns if "lon" in c and "min" not in c and "max" not in c), None)

        if faltan_lum.any() and col_lat and col_lon:
            coords_m = df_m[[col_lat, col_lon]].apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", "."), errors="coerce")).to_numpy()
            lums_m = df_m["lum_num"].to_numpy()
            valid_m = ~np.isnan(coords_m).any(axis=1)
            coords_m, lums_m = coords_m[valid_m], lums_m[valid_m]

            coords_g = df_grid.loc[faltan_lum, ["centro_lat", "centro_lon"]].to_numpy()
            dists = ((coords_g[:, None, :] - coords_m[None, :, :]) ** 2).sum(axis=2)
            df_grid.loc[faltan_lum, "luminosidad_promedio"] = lums_m[dists.argmin(axis=1)]

        df_grid["luminosidad_promedio"] = df_grid["luminosidad_promedio"].fillna(75.43).round(2)
        Actor.log.info(
            f"Luminosidad asignada desde '{col_lum}' -> min: {df_grid['luminosidad_promedio'].min()}, "
            f"max: {df_grid['luminosidad_promedio'].max()}, "
            f"valores únicos: {df_grid['luminosidad_promedio'].nunique()}"
        )
    else:
        Actor.log.warning("No se detectó columna de luminosidad en el maestro; usando 75.43.")
        df_grid["luminosidad_promedio"] = 75.43

    return df_grid.to_dict(orient="records")


COLUMNAS_ESPERADAS_CLEAN = [
    "zona_id", "lat_min", "lat_max", "lon_min", "lon_max",
    "centro_lat", "centro_lon", "delitos_nocturnos",
    "proporcion_fin_semana", "hora_predominante",
    "tipo_predominante", "luminosidad_promedio"
]


def ejecutar_eda(clean_records, df_maestro: pd.DataFrame, eda_tabla: str) -> list:
    if isinstance(clean_records, str):
        clean_records = json.loads(clean_records)

    df_zonas = pd.DataFrame(clean_records) if clean_records else df_maestro.copy()
    df_zonas.columns = [str(c).strip().lower() for c in df_zonas.columns]

    # Si Make envió col1, col2... en lugar de los nombres reales, los renombramos por posición
    if "delitos_nocturnos" not in df_zonas.columns and "col8" in df_zonas.columns:
        mapeo_clean = {
            f"col{i + 1}": nombre
            for i, nombre in enumerate(COLUMNAS_ESPERADAS_CLEAN)
            if f"col{i + 1}" in df_zonas.columns
        }
        df_zonas = df_zonas.rename(columns=mapeo_clean)

    # Eliminar fila de encabezados si quedó incluida como dato
    if "zona_id" in df_zonas.columns:
        df_zonas = df_zonas[df_zonas["zona_id"].astype(str).str.lower() != "zona_id"].copy()

    if "zona_id_norm" in df_zonas.columns and "zona_id" not in df_zonas.columns:
        df_zonas["zona_id"] = df_zonas["zona_id_norm"]

    if "delitos_nocturnos" not in df_zonas.columns:
        raise ValueError(f"Falta 'delitos_nocturnos'. Columnas recibidas en EDA: {df_zonas.columns.tolist()}")

    df_zonas["delitos_nocturnos"] = pd.to_numeric(df_zonas["delitos_nocturnos"], errors="coerce").fillna(0).astype(int)
    col_lum_eda = next((c for c in df_zonas.columns if "lum" in c), None)
    if col_lum_eda:
        df_zonas["luminosidad_promedio"] = pd.to_numeric(
            df_zonas[col_lum_eda].astype(str).str.replace(",", "."), errors="coerce"
        ).fillna(75.43)
    else:
        df_zonas["luminosidad_promedio"] = 75.43
    
    df_zonas["nivel_incidencia"] = "Sin registros nocturnos"
    mask_pos = df_zonas["delitos_nocturnos"] > 0
    if mask_pos.any():
        q33 = df_zonas.loc[mask_pos, "delitos_nocturnos"].quantile(1 / 3)
        q67 = df_zonas.loc[mask_pos, "delitos_nocturnos"].quantile(2 / 3)
        df_zonas.loc[mask_pos & (df_zonas["delitos_nocturnos"] <= q33), "nivel_incidencia"] = "Bajo"
        df_zonas.loc[mask_pos & (df_zonas["delitos_nocturnos"] > q33) & (df_zonas["delitos_nocturnos"] <= q67), "nivel_incidencia"] = "Medio"
        df_zonas.loc[mask_pos & (df_zonas["delitos_nocturnos"] > q67), "nivel_incidencia"] = "Alto"

    df_zonas["clasificacion_luminosidad"] = df_zonas["luminosidad_promedio"].apply(clasificar_luminosidad)

    csv_1 = (
        df_zonas.groupby("nivel_incidencia", as_index=False)
        .agg(
            cantidad_celdas=("zona_id", "count"),
            total_delitos_nocturnos=("delitos_nocturnos", "sum"),
            luminosidad_media=("luminosidad_promedio", "mean"),
        )
    )
    csv_1["luminosidad_media"] = csv_1["luminosidad_media"].round(2)
    csv_1["tabla_origen"] = "csv_1_incidencia"

    csv_2 = (
        df_zonas.sort_values("delitos_nocturnos", ascending=False)
        .head(15)[["zona_id", "delitos_nocturnos", "luminosidad_promedio", "nivel_incidencia"]]
        .copy()
    )
    csv_2["tabla_origen"] = "csv_2_top_zonas"

    csv_3 = (
        df_zonas.groupby("clasificacion_luminosidad", as_index=False)
        .agg(
            cantidad_celdas=("zona_id", "count"),
            promedio_delitos=("delitos_nocturnos", "mean"),
            total_delitos=("delitos_nocturnos", "sum"),
        )
    )
    csv_3["promedio_delitos"] = csv_3["promedio_delitos"].round(2)
    csv_3["tabla_origen"] = "csv_3_luminosidad"

    if eda_tabla == "incidencia":
        return csv_1.to_dict(orient="records")
    elif eda_tabla == "top_zonas":
        return csv_2.to_dict(orient="records")
    elif eda_tabla == "luminosidad":
        return csv_3.to_dict(orient="records")

    salida_unificada = []
    for df_part in [csv_1, csv_2, csv_3]:
        salida_unificada.extend(df_part.to_dict(orient="records"))
    return salida_unificada


def cargar_modelo_ml():
    for raiz, _, archivos in os.walk("/usr/src/app"):
        for archivo in archivos:
            if archivo == "modelo_riesgo_caba.json":
                ruta = os.path.join(raiz, archivo)
                with open(ruta, "r", encoding="utf-8") as f:
                    data = json.load(f)
                Actor.log.info(f"Modelo ML cargado desde JSON puro: {ruta}")
                return data
    Actor.log.warning("No se encontró modelo_riesgo_caba.json; se usará heurística de respaldo.")
    return None


def predecir_con_bosque_json(artefacto: dict, vector_x: list) -> tuple[str, float]:
    clases = artefacto["clases"]
    probas_acum = np.zeros(len(clases), dtype=float)

    for arbol in artefacto["arboles"]:
        nodo = 0
        while arbol["children_left"][nodo] != -1:
            feat_idx = arbol["feature"][nodo]
            if vector_x[feat_idx] <= arbol["threshold"][nodo]:
                nodo = arbol["children_left"][nodo]
            else:
                nodo = arbol["children_right"][nodo]
        probas_acum += np.array(arbol["value"][nodo], dtype=float)

    probas = probas_acum / len(artefacto["arboles"])
    idx_max = int(np.argmax(probas))
    return str(clases[idx_max]), round(float(probas[idx_max]), 4)


def ejecutar_inference(actor_input: dict, df_zonas: pd.DataFrame) -> list:
    consultas = actor_input.get("bulk_items")
    if isinstance(consultas, str):
        consultas = json.loads(consultas)

    if not consultas:
        consultas = [{
            "zona_id": actor_input.get("zona_id", "-34.5880_-58.4160"),
            "dia_semana": actor_input.get("dia_semana", "Viernes"),
            "franja_horaria": int(actor_input.get("franja_horaria", 22)),
        }]

    artefacto = cargar_modelo_ml()
    modelo_nombre = artefacto.get("modelo_nombre", "HeuristicaFallback") if artefacto else "HeuristicaFallback"

    col_lum = next((c for c in df_zonas.columns if "lum" in c), None) if not df_zonas.empty else None

    def norm_zid(z):
        try:
            lat_s, lon_s = str(z).strip().split("_")
            return f"{float(lat_s):.4f}_{float(lon_s):.4f}"
        except Exception:
            return str(z).strip()

    if not df_zonas.empty and "zona_id_norm" in df_zonas.columns:
        df_zonas = df_zonas.copy()
        df_zonas["zona_id_4d"] = df_zonas["zona_id_norm"].apply(norm_zid)

    dias_fds = {"sabado", "sábado", "domingo", "saturday", "sunday", "5", "6"}
    resultados = []

    for item in consultas:
        raw_zid = str(item.get("zona_id") or item.get("col1") or "").strip()
        if not raw_zid or raw_zid.lower() == "zona_id":
            continue

        dia_input = str(item.get("dia_semana") or item.get("col2") or "Viernes").strip()
        hora_input = int(float(str(item.get("franja_horaria") or item.get("col3") or 22).strip()))
        es_fds = 1 if dia_input.lower() in dias_fds else 0

        try:
            lat_str, lon_str = raw_zid.split("_")
            lat_raw, lon_raw = float(lat_str), float(lon_str)
            # Snap automático a la celda de la grilla que contiene ese punto
            gi = int(np.floor((lat_raw - LAT_MIN + 1e-6) / TAM_LAT))
            gj = int(np.floor((lon_raw - LON_MIN + 1e-6) / TAM_LON))
            z_lat = round(LAT_MIN + gi * TAM_LAT, 4)
            z_lon = round(LON_MIN + gj * TAM_LON, 4)
            zona_input = f"{z_lat:.4f}_{z_lon:.4f}"
            latitud, longitud = round(z_lat + TAM_LAT / 2, 6), round(z_lon + TAM_LON / 2, 6)
        except Exception:
            zona_input = norm_zid(raw_zid)
            lat_raw, lon_raw = -34.6037, -58.3816
            latitud, longitud = lat_raw, lon_raw

        # Búsqueda espacial directa por bounding box usando el centro de la celda (latitud, longitud)
        fila_zona = pd.DataFrame()
        if not df_zonas.empty and all(c in df_zonas.columns for c in ["lat_min", "lat_max", "lon_min", "lon_max"]):
            for col_c in ["lat_min", "lat_max", "lon_min", "lon_max"]:
                df_zonas[col_c] = pd.to_numeric(df_zonas[col_c].astype(str).str.replace(",", "."), errors="coerce")

            cond_espacial = (
                (df_zonas["lat_min"] <= latitud) & (latitud <= df_zonas["lat_max"]) &
                (df_zonas["lon_min"] <= longitud) & (longitud <= df_zonas["lon_max"])
            )
            fila_zona = df_zonas[cond_espacial]
            
        if not fila_zona.empty:
            datos = fila_zona.iloc[0]
            lum_val = pd.to_numeric(str(datos[col_lum]).replace(",", "."), errors="coerce") if col_lum else np.nan
            luminosidad = round(float(lum_val), 2) if pd.notna(lum_val) else 75.43
            delitos_hist = int(float(str(datos["delitos_nocturnos"]).replace(",", ".")))
            nivel_base = str(datos["nivel_incidencia"]).strip()
            zona_encontrada = True
        else:
            luminosidad = 75.43
            delitos_hist = 0
            nivel_base = "Bajo"
            zona_encontrada = False

        if artefacto is not None and "arboles" in artefacto:
            vector_x = [latitud, longitud, luminosidad, hora_input, es_fds]
            nivel_estimado, confianza = predecir_con_bosque_json(artefacto, vector_x)
        else:
            nivel_estimado = predecir_riesgo_temporal(nivel_base, hora_input)
            confianza = 0.75

        resultados.append({
            "zona_id": zona_input,
            "latitud": latitud,
            "longitud": longitud,
            "zona_valida_en_dataset": zona_encontrada,
            "dia_semana": dia_input,
            "franja_horaria": hora_input,
            "nivel_incidencia_estimado": nivel_estimado,
            "confianza_modelo": confianza,
            "modelo_utilizado": modelo_nombre,
            "luminosidad_promedio": luminosidad,
            "clasificacion_luminosidad": clasificar_luminosidad(luminosidad),
            "delitos_nocturnos_historicos": delitos_hist,
            "patron_historico": f"Registro histórico de {delitos_hist} delitos nocturnos en la celda (incidencia base: {nivel_base})",
        })

    return resultados


def cargar_dataset_maestro() -> pd.DataFrame:
    for raiz, _, archivos in os.walk("/usr/src/app"):
        for archivo in archivos:
            if archivo == "dataset_inteligencia_territorial.csv":
                ruta = os.path.join(raiz, archivo)
                # Detecta automáticamente si el CSV usa coma o punto y coma
                df = pd.read_csv(ruta, sep=None, engine="python")
                df.columns = [str(c).strip().lower() for c in df.columns]
                Actor.log.info(f"Dataset maestro cargado ({len(df)} filas) | Columnas: {df.columns.tolist()}")
                if "zona_id" in df.columns:
                    df["zona_id_norm"] = df["zona_id"].astype(str).str.strip()
                return df
    Actor.log.warning("NO se encontró dataset_inteligencia_territorial.csv en /usr/src/app")
    return pd.DataFrame()


async def main() -> None:
    async with Actor:
        try:
            actor_input = await Actor.get_input() or {}
            operation = str(actor_input.get("operation", "inference")).strip().lower()
            df_maestro = cargar_dataset_maestro()

            if operation == "etl":
                Actor.log.info("Ejecutando Producto 1 - Etapa ETL...")
                raw_records = actor_input.get("raw_records", [])
                resultados = ejecutar_etl(raw_records, df_maestro)

            elif operation == "eda":
                Actor.log.info("Ejecutando Producto 1 - Etapa EDA...")
                clean_records = actor_input.get("clean_records", [])
                eda_tabla = str(actor_input.get("eda_tabla", "all")).strip().lower()
                resultados = ejecutar_eda(clean_records, df_maestro, eda_tabla)

            else:
                Actor.log.info("Ejecutando Producto 2 - Etapa Inference...")
                resultados = ejecutar_inference(actor_input, df_maestro)

            await Actor.push_data(resultados)
            Actor.log.info(f"Operación '{operation}' finalizada con {len(resultados)} registros.")

        except Exception as e:
            Actor.log.error(f"Error en main(): {e}")
            Actor.log.error(traceback.format_exc())
            raise