# 🌃 Data Horizon — Inteligencia Territorial Nocturna (CABA)

Plataforma integral de analítica geoespacial y predicción de riesgo delictivo nocturno en la Ciudad Autónoma de Buenos Aires (CABA), desarrollada mediante la integración de **datos abiertos de delitos (GCBA)** e **imágenes satelitales de luminosidad nocturna (NASA Black Marble)**.

El sistema combina un motor de procesamiento e inferencia en la nube sobre **Apify**, automatización de flujos con **Make** y **Google Drive**, y una aplicación web interactiva construida en **Streamlit** para visualización ejecutiva y toma de decisiones preventivas.

---

## 🏗️ Arquitectura de la Solución

El proyecto implementa los dos productos requeridos en un flujo unificado de punta a punta:

1. **Producto 1 — Analytics (`ETL + EDA`):** Extracción, limpieza, cruce espacial por grilla territorial de **1.600 celdas** de CABA y análisis exploratorio de distribución delictiva y niveles de luminosidad satelital.
2. **Producto 2 — Inference (`RandomForest`):** Servicio predictivo que evalúa el nivel de riesgo nocturno (`Bajo`, `Medio`, `Alto`) y su porcentaje de confianza combinando ubicación geográfica, luminosidad satelital, día de la semana y franja horaria.
3. **Dashboard Interactivo (`Data Horizon`):** Interfaz web desplegada en **Streamlit Community Cloud** con mapa interactivo de CABA (`OpenStreetMap`), consulta por clic directo sobre celdas, simulación temporal (día y hora) y vista ejecutiva de indicadores macro.

### Estructura del Repositorio

```text
├── actor_apify/
│   ├── .actor/                 # Esquemas de configuración, entrada y salida de Apify
│   ├── my_actor/               # Código fuente del Actor (main.py), grilla y modelo JSON
│   ├── Dockerfile              # Definición del contenedor del servicio
│   └── requirements.txt        # Dependencias del Actor
├── colabs/
│   ├── analytics.ipynb         # Notebook de ETL, cruce geoespacial y EDA
│   └── training.ipynb          # Notebook de entrenamiento, validación y exportación del modelo
├── dashboard/
│   └── app.py                  # Aplicación web interactiva en Streamlit
├── data/
│   ├── raw/                    # Datasets originales sin procesar
│   └── processed/              # Dataset maestro consolidado (dataset_inteligencia_territorial.csv)
├── requirements.txt            # Dependencias del Dashboard web
└── README.md
```

---

## ⚙️ Parámetros de Entrada del Servicio (`Input`)

El Actor de Apify admite ejecución mediante el parámetro `operation` (`etl`, `eda` o `inference`), soportando tanto **Consulta Individual** como **Consulta por Lotes (`bulk_items`)**:

| Parámetro | Tipo | Requerido | Valor por defecto | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| `operation` | `string` | No | `"inference"` | Operación a ejecutar: `"etl"`, `"eda"` o `"inference"` (predicción ML). |
| `zona_id` | `string` | No* | `"-34.5880_-58.4160"` | Identificador de la celda geográfica (`latitud_longitud`) en CABA. |
| `dia_semana` | `string` | No | `"Viernes"` | Día de la semana a evaluar (`Lunes` a `Domingo`). |
| `franja_horaria` | `integer` | No | `21` | Hora del día en formato 24 hs (`0` a `23`). |
| `bulk_items` | `array` | No | `[]` | Lista de objetos JSON para evaluación masiva. Tiene prioridad sobre los campos individuales. |

*\*Si se consulta una coordenada fuera de las celdas activas del dataset maestro, el motor aplica valores de referencia promedio de CABA e informa `zona_valida_en_dataset: false`.*

---

## 🚀 Ejemplos de Uso (API POST)

### 1. Consulta Individual (`inference`)
```json
{
  "operation": "inference",
  "zona_id": "-34.5880_-58.4160",
  "dia_semana": "Sabado",
  "franja_horaria": 2
}
```

### 2. Consulta por Lotes (`bulk_items`)
```json
{
  "operation": "inference",
  "bulk_items": [
    {
      "zona_id": "-34.5880_-58.4160",
      "dia_semana": "Viernes",
      "franja_horaria": 21
    },
    {
      "zona_id": "-34.6037_-58.3816",
      "dia_semana": "Sabado",
      "franja_horaria": 3
    }
  ]
}
```

---

## 📤 Formato de Salida (`Output Dataset`)

Por cada consulta procesada, el servicio devuelve un objeto JSON estructurado con la predicción del modelo y los indicadores territoriales asociados:

```json
[
  {
    "zona_id": "-34.5880_-58.4160",
    "latitud": -34.5880,
    "longitud": -58.4160,
    "zona_valida_en_dataset": true,
    "dia_semana": "Sabado",
    "franja_horaria": 2,
    "nivel_incidencia_estimado": "Alto",
    "confianza_modelo": 0.914,
    "modelo_utilizado": "RandomForest",
    "luminosidad_promedio": 78.45,
    "clasificacion_luminosidad": "Media",
    "delitos_nocturnos_historicos": 42,
    "patron_historico": "Registro histórico de 42 delitos nocturnos en la celda (incidencia base: Alto)"
  }
]
```

### Descripción de campos de salida
* **`latitud` / `longitud`:** Coordenadas del centroide de la celda para renderizado cartográfico.
* **`zona_valida_en_dataset`:** Indica si la celda cuenta con relevamiento directo en la grilla de CABA (`true`) o utiliza respaldo estadístico (`false`).
* **`nivel_incidencia_estimado`:** Categoría de riesgo estimada (`Bajo`, `Medio`, `Alto`) combinando el modelo `RandomForest` con el ajuste dinámico por franja horaria y fin de semana.
* **`confianza_modelo` / `modelo_utilizado`:** Probabilidad asociada a la clase predicha e identificador del modelo entrenado.
* **`luminosidad_promedio` / `clasificacion_luminosidad`:** Índice de radiancia nocturna (nW) y su categorización (`Baja`, `Media`, `Alta`).
* **`delitos_nocturnos_historicos` / `patron_historico`:** Conteo histórico de delitos nocturnos registrados en la celda y resumen descriptivo.
