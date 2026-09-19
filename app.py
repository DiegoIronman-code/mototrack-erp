"""
Motor de Pronosticos MotoTrack
------------------------------
App en Streamlit que carga el historico de demanda, compara varios modelos
de pronostico mediante backtesting y muestra el mejor modelo por serie.

Ademas incluye una segunda pestana para clasificar los SKUs en ABC (por
utilidad) y XYZ (por Score% del pronostico), a partir de un archivo de
costos y precios que se conserva entre sesiones.
"""

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla/navegador, sirve en cualquier servidor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from statsforecast import StatsForecast
from statsforecast.models import (
    HoltWinters,
    MSTL,
    SimpleExponentialSmoothingOptimized,
    WindowAverage,
)

# ---------------------------------------------------------------------------
# Configuracion general
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ERP MotoTrack", page_icon=":material/factory:", layout="wide"
)
st.logo(str(Path(__file__).resolve().parent / "assets" / "logo_mototrack.svg"), size="large")

FREQ = "W-MON"
FECHA_BASE = pd.Timestamp("2023-01-02")  # lunes de referencia para las fechas ficticias
# Colores de marca (inspirados en Universidad EAFIT: azul institucional + acentos)
COLORES_PRODUCTO = {"MOTO": "#000C66", "CUATRIMOTO": "#00A9E0", "TRACTOR": "#FF9900"}
REGIONALES_BASE = ["NORTE", "CENTRO", "SUR"]

BLOQUES_A_REGIONAL = {0: "NORTE", 1: "CENTRO", 2: "SUR"}

VENTANA_SEMANAS = 52  # ventana movil de un anio para el pronostico

REGIONALES_SKU = ["NORTE", "CENTRO", "SUR"]  # sedes reales; CEDI y MOTOTRAK son agregados, no SKU
VENTANA_UTILIDAD_DEFAULT = 52 * 4  # 4 anios de historia para acumular la utilidad
PRECIOS_VENTA_DEFAULT = {"MOTO": 7_000_000, "CUATRIMOTO": 9_000_000, "TRACTOR": 11_000_000}

TASA_INVENTARIO_DEFAULT = 17.0  # % EA, tasa de costo de mantener inventario

ABC_CORTE_A_DEFAULT = 80.0  # % acumulado de utilidad hasta el cual un SKU es clase A
ABC_CORTE_B_DEFAULT = 95.0  # % acumulado de utilidad hasta el cual un SKU es clase B

# Umbrales XYZ por Score%, segun el material del curso "Gestion de Inventarios"
# (Dhoka & Choudary, 2013): X <= 25%, 25% < Y <= 60%, Z > 60%.
XYZ_CORTE_X_DEFAULT = 25.0
XYZ_CORTE_Y_DEFAULT = 60.0

MOSTRAR_TAB_EOQ = False  # pestana de EOQ oculta por ahora; poner en True para volver a mostrarla
COLORES_A_PROVEEDOR = {"AZUL": "AZUL", "AMARILLO": "AMARILLO", "NEGRO": "NEGRO"}  # GRIS no se usa en este ejercicio

# Carpeta donde se guarda el archivo de costos y los precios entre sesiones
# (sin base de datos: son archivos planos que se sobreescriben al actualizar).
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
COSTOS_PATH = DATA_DIR / "costos_actual.xlsx"
PRECIOS_PATH = DATA_DIR / "precios_venta.json"


def construir_modelos():
    """Crea la lista fija de modelos a evaluar (definida en el ejercicio de clase)."""
    return [
        HoltWinters(season_length=13, alias="hw"),
        WindowAverage(window_size=3, alias="wa_3"),
        WindowAverage(window_size=6, alias="wa_6"),
        WindowAverage(window_size=12, alias="wa_12"),
        SimpleExponentialSmoothingOptimized(alias="ses"),
        MSTL(season_length=13, alias="mstl"),
    ]


# ---------------------------------------------------------------------------
# 1. Carga y transformacion de datos de demanda
# ---------------------------------------------------------------------------

def cargar_datos(archivo) -> pd.DataFrame:
    """Lee la hoja Demand y la convierte a formato largo:
    Turn | REGIONAL | PRODUCTO | DEMANDA
    """
    df_raw = pd.read_excel(archivo, sheet_name="Demand", header=1)

    # Separamos los bloques horizontales usando las columnas vacias (Unnamed) como frontera
    columnas = list(df_raw.columns)
    bloques = []
    actual = []
    for col in columnas:
        if str(col).startswith("Unnamed"):
            if actual:
                bloques.append(actual)
                actual = []
        else:
            actual.append(col)
    if actual:
        bloques.append(actual)

    if len(bloques) != 3:
        raise ValueError(
            f"Se esperaban 3 bloques de datos (Norte, Centro, Sur) y se encontraron {len(bloques)}."
        )

    partes = []
    for i, bloque in enumerate(bloques):
        regional = BLOQUES_A_REGIONAL[i]
        col_turn = bloque[0]
        cols_producto = bloque[1:]

        sub = df_raw[[col_turn] + cols_producto].copy()
        sub = sub.rename(columns={col_turn: "Turn"})
        # Quitamos el sufijo que pandas agrega a columnas duplicadas (ej. "MOTO.1")
        sub.columns = ["Turn"] + [str(c).split(".")[0] for c in cols_producto]

        largo = sub.melt(id_vars="Turn", var_name="PRODUCTO", value_name="DEMANDA")
        largo = largo.dropna(subset=["DEMANDA"])
        largo["REGIONAL"] = regional
        partes.append(largo)

    df_long = pd.concat(partes, ignore_index=True)
    df_long["Turn"] = df_long["Turn"].astype(int)
    df_long = df_long[["Turn", "REGIONAL", "PRODUCTO", "DEMANDA"]]

    # Regionales calculadas: CEDI = CENTRO + SUR, MOTOTRAK = NORTE + CENTRO + SUR
    cedi = (
        df_long[df_long["REGIONAL"].isin(["CENTRO", "SUR"])]
        .groupby(["Turn", "PRODUCTO"], as_index=False)["DEMANDA"]
        .sum()
    )
    cedi["REGIONAL"] = "CEDI"

    mototrak = (
        df_long.groupby(["Turn", "PRODUCTO"], as_index=False)["DEMANDA"].sum()
    )
    mototrak["REGIONAL"] = "MOTOTRAK"

    df_long = pd.concat([df_long, cedi, mototrak], ignore_index=True)
    return df_long


def aplicar_ventana_movil(df_long: pd.DataFrame, semanas: int = VENTANA_SEMANAS) -> pd.DataFrame:
    """Se queda solo con las 'semanas' mas recientes del historico (segun Turn).

    Cada archivo de demanda que se sube trae el historico completo hasta la
    fecha; al aplicar esta ventana, las semanas mas antiguas quedan
    descartadas automaticamente cada vez que llega un turno nuevo.
    """
    turno_max = int(df_long["Turn"].max())
    turno_desde = turno_max - semanas + 1
    return df_long[df_long["Turn"] >= turno_desde].reset_index(drop=True)


def a_formato_statsforecast(df_long: pd.DataFrame) -> pd.DataFrame:
    """Convierte el formato largo al esperado por StatsForecast: unique_id, ds, y."""
    df_ts = df_long.copy()
    df_ts["unique_id"] = df_ts["REGIONAL"] + "|" + df_ts["PRODUCTO"]
    df_ts["ds"] = FECHA_BASE + pd.to_timedelta((df_ts["Turn"] - 1) * 7, unit="D")
    df_ts["y"] = df_ts["DEMANDA"].astype(float)
    return df_ts[["unique_id", "Turn", "ds", "y"]].sort_values(["unique_id", "ds"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Backtesting y seleccion del mejor modelo
# ---------------------------------------------------------------------------

def ejecutar_backtesting(df_ts: pd.DataFrame, n_windows: int, h: int):
    """Corre cross_validation modelo por modelo y serie por serie, para poder
    descartar de forma aislada cualquier combinacion que falle."""
    modelos = construir_modelos()
    resultados_cv = []
    errores = []

    for unique_id, serie in df_ts.groupby("unique_id"):
        serie = serie[["unique_id", "ds", "y"]].sort_values("ds").reset_index(drop=True)
        piezas = []
        for modelo in modelos:
            try:
                sf = StatsForecast(models=[modelo], freq=FREQ, n_jobs=1)
                cv = sf.cross_validation(df=serie, h=h, n_windows=n_windows, step_size=h)
                piezas.append(cv)
            except Exception as e:
                errores.append(f"Serie '{unique_id}', modelo '{modelo.alias}': {e}")

        if not piezas:
            continue

        combinado = piezas[0]
        for extra in piezas[1:]:
            columnas_nuevas = [c for c in extra.columns if c not in combinado.columns]
            combinado = combinado.merge(
                extra[["unique_id", "ds", "cutoff"] + columnas_nuevas],
                on=["unique_id", "ds", "cutoff"],
                how="outer",
            )
        resultados_cv.append(combinado)

    if not resultados_cv:
        return pd.DataFrame(), errores, modelos

    df_cv = pd.concat(resultados_cv, ignore_index=True)
    return df_cv, errores, modelos


def calcular_metricas(df_cv: pd.DataFrame) -> pd.DataFrame:
    """Calcula MAE%, Sesgo%, Score% y RMSE por serie y por modelo."""
    if df_cv.empty:
        return pd.DataFrame()

    columnas_meta = {"unique_id", "ds", "cutoff", "y"}
    columnas_modelo = [c for c in df_cv.columns if c not in columnas_meta]

    filas = []
    for unique_id, grupo in df_cv.groupby("unique_id"):
        for modelo in columnas_modelo:
            sub = grupo.dropna(subset=[modelo])
            if sub.empty:
                continue
            y = sub["y"].to_numpy(dtype=float)
            pred = sub[modelo].to_numpy(dtype=float)
            suma_y = y.sum()
            if suma_y == 0:
                continue

            mae_pct = np.sum(np.abs(y - pred)) / suma_y
            sesgo_pct = np.sum(y - pred) / suma_y
            score_pct = mae_pct + abs(sesgo_pct)
            rmse = np.sqrt(np.mean((y - pred) ** 2))

            filas.append(
                {
                    "unique_id": unique_id,
                    "MODELO": modelo,
                    "MAE_PCT": mae_pct,
                    "SESGO_PCT": sesgo_pct,
                    "SCORE_PCT": score_pct,
                    "RMSE": rmse,
                }
            )

    return pd.DataFrame(filas)


def elegir_mejor_modelo(df_metricas: pd.DataFrame) -> pd.DataFrame:
    """Se queda con el modelo de menor Score% para cada serie."""
    if df_metricas.empty:
        return df_metricas
    idx = df_metricas.groupby("unique_id")["SCORE_PCT"].idxmin()
    return df_metricas.loc[idx].reset_index(drop=True)


def generar_pronostico_final(df_ts: pd.DataFrame, ganadores: pd.DataFrame, modelos, h: int):
    """Genera el pronostico de h periodos hacia adelante con el mejor modelo de cada serie."""
    modelos_por_alias = {m.alias: m for m in modelos}
    piezas = []
    errores = []

    for _, fila in ganadores.iterrows():
        unique_id = fila["unique_id"]
        alias = fila["MODELO"]
        modelo = modelos_por_alias.get(alias)
        if modelo is None:
            continue

        serie = df_ts[df_ts["unique_id"] == unique_id].sort_values("ds").reset_index(drop=True)
        ultimo_turno = int(serie["Turn"].max())

        try:
            sf = StatsForecast(models=[modelo], freq=FREQ, n_jobs=1)
            fcst = sf.forecast(df=serie[["unique_id", "ds", "y"]], h=h)
            fcst = fcst.sort_values("ds").reset_index(drop=True)
            fcst["Turn"] = range(ultimo_turno + 1, ultimo_turno + h + 1)
            fcst["VALOR"] = fcst[alias]
            piezas.append(fcst[["unique_id", "Turn", "ds", "VALOR"]])
        except Exception as e:
            errores.append(f"Pronostico final, serie '{unique_id}', modelo '{alias}': {e}")

    if not piezas:
        return pd.DataFrame(), errores
    return pd.concat(piezas, ignore_index=True), errores


# ---------------------------------------------------------------------------
# 3. Tabla resumen
# ---------------------------------------------------------------------------

def construir_tabla_resumen(ganadores: pd.DataFrame, df_forecast: pd.DataFrame) -> pd.DataFrame:
    filas = []
    turnos_futuros = sorted(df_forecast["Turn"].unique()) if not df_forecast.empty else []

    for _, fila in ganadores.iterrows():
        unique_id = fila["unique_id"]
        regional, producto = unique_id.split("|")
        registro = {
            "REGIONAL": regional,
            "PRODUCTO": producto,
            "MODELO": fila["MODELO"],
            "SCORE_PORC": round(fila["SCORE_PCT"] * 100, 1),
            "RMSE": round(fila["RMSE"], 2),
        }

        serie_fcst = df_forecast[df_forecast["unique_id"] == unique_id]
        for turno in turnos_futuros:
            valor = serie_fcst.loc[serie_fcst["Turn"] == turno, "VALOR"]
            registro[f"Turno {turno}"] = round(valor.iloc[0], 1) if not valor.empty else np.nan

        filas.append(registro)

    return pd.DataFrame(filas)


def df_a_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Resumen")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 4. Grafica
# ---------------------------------------------------------------------------

def construir_grafica(df_long: pd.DataFrame, df_forecast: pd.DataFrame, ganadores: pd.DataFrame, h: int):
    titulos = ["NORTE", "CENTRO", "SUR", "CEDI (C+S)", "MOTOTRAK (N+C+S)", ""]
    posiciones = {
        "NORTE": (1, 1),
        "CENTRO": (1, 2),
        "SUR": (1, 3),
        "CEDI": (2, 1),
        "MOTOTRAK": (2, 2),
    }

    fig = make_subplots(rows=2, cols=3, subplot_titles=titulos)
    leyenda_mostrada = set()

    ganadores_por_id = ganadores.set_index("unique_id")["MODELO"].to_dict() if not ganadores.empty else {}
    max_turno_global = int(df_long["Turn"].max())
    turno_desde = max_turno_global - (52 + h) + 1

    for regional, (fila, col) in posiciones.items():
        datos_regional = df_long[(df_long["REGIONAL"] == regional) & (df_long["Turn"] >= turno_desde)]

        for producto in ["MOTO", "CUATRIMOTO", "TRACTOR"]:
            datos_producto = datos_regional[datos_regional["PRODUCTO"] == producto].sort_values("Turn")
            if datos_producto.empty:
                continue

            color = COLORES_PRODUCTO[producto]
            mostrar_leyenda = producto not in leyenda_mostrada
            leyenda_mostrada.add(producto)

            # Linea solida: demanda real
            fig.add_trace(
                go.Scatter(
                    x=datos_producto["Turn"],
                    y=datos_producto["DEMANDA"],
                    mode="lines",
                    name=producto,
                    legendgroup=producto,
                    showlegend=mostrar_leyenda,
                    line=dict(color=color),
                ),
                row=fila,
                col=col,
            )

            # Linea punteada: pronostico del mejor modelo
            unique_id = f"{regional}|{producto}"
            if unique_id in ganadores_por_id and not df_forecast.empty:
                datos_fcst = df_forecast[df_forecast["unique_id"] == unique_id].sort_values("Turn")
                if not datos_fcst.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=datos_fcst["Turn"],
                            y=datos_fcst["VALOR"],
                            mode="lines",
                            name=producto,
                            legendgroup=producto,
                            showlegend=False,
                            line=dict(color=color, dash="dot"),
                        ),
                        row=fila,
                        col=col,
                    )

    fig.update_layout(template="ggplot2", height=700)
    fig.update_xaxes(title_text="Turn")
    fig.update_yaxes(title_text="Demanda")
    return fig


# ---------------------------------------------------------------------------
# 5. Costos, precios y clasificacion ABC / XYZ
# ---------------------------------------------------------------------------

def guardar_costos(archivo) -> None:
    """Persiste el Excel de costos en disco para que quede disponible en la
    siguiente sesion aunque no se vuelva a subir."""
    COSTOS_PATH.write_bytes(archivo.getvalue())


def extraer_costo_por_producto(ruta_o_archivo) -> pd.DataFrame:
    """Busca en la hoja 'Costos' el bloque 'Producto | ... | Costo Total
    Unitario' (seccion 5, "Resumen - Costo Unitario de Fabricacion") y
    devuelve PRODUCTO -> COSTO_UNITARIO.

    No depende de un numero de fila fijo, solo de encontrar una fila cuya
    primera celda sea "Producto" y que tenga una columna "Costo Total ...".
    """
    df_raw = pd.read_excel(ruta_o_archivo, sheet_name="Costos", header=None)

    fila_header = None
    col_costo = None
    for i in range(len(df_raw)):
        fila = df_raw.iloc[i]
        primera = str(fila.iloc[0]).strip().lower()
        if primera == "producto":
            for j, valor in enumerate(fila):
                if "costo total" in str(valor).strip().lower():
                    fila_header = i
                    col_costo = j
                    break
            if fila_header is not None:
                break

    if fila_header is None:
        raise ValueError(
            "No se encontro en la hoja 'Costos' una fila 'Producto' con una "
            "columna 'Costo Total Unitario'."
        )

    filas = []
    for i in range(fila_header + 1, len(df_raw)):
        nombre = df_raw.iloc[i, 0]
        costo = df_raw.iloc[i, col_costo]
        if pd.isna(nombre) or pd.isna(costo):
            break
        filas.append({"PRODUCTO": str(nombre).strip().upper(), "COSTO_UNITARIO": float(costo)})

    if not filas:
        raise ValueError("El bloque 'Costo Total Unitario' no tiene filas de datos.")

    return pd.DataFrame(filas)


def cargar_costos_guardados() -> pd.DataFrame:
    """Lee el costo unitario por producto del Excel de costos persistido en disco."""
    return extraer_costo_por_producto(COSTOS_PATH)


def cargar_precios() -> dict:
    """Lee los precios de venta guardados en disco; si no existen, usa los
    valores por defecto."""
    if PRECIOS_PATH.exists():
        with open(PRECIOS_PATH, "r", encoding="utf-8") as f:
            guardados = json.load(f)
        return {**PRECIOS_VENTA_DEFAULT, **guardados}
    return dict(PRECIOS_VENTA_DEFAULT)


def guardar_precios(precios: dict) -> None:
    """Persiste los precios de venta editables en disco."""
    with open(PRECIOS_PATH, "w", encoding="utf-8") as f:
        json.dump(precios, f, ensure_ascii=False, indent=2)


def calcular_utilidad_por_sku(
    df_long_completo: pd.DataFrame,
    costos_por_producto: pd.DataFrame,
    precios_por_producto: dict,
    ventana_semanas: int,
) -> pd.DataFrame:
    """Utilidad total acumulada de las ultimas 'ventana_semanas' de demanda
    real, por SKU (REGIONAL|PRODUCTO), a partir de precio y costo unitario
    por producto (iguales en las 3 sedes)."""
    turno_max = int(df_long_completo["Turn"].max())
    turno_desde = turno_max - ventana_semanas + 1
    ventana = df_long_completo[
        (df_long_completo["Turn"] >= turno_desde)
        & (df_long_completo["REGIONAL"].isin(REGIONALES_SKU))
    ]

    cantidades = ventana.groupby(["REGIONAL", "PRODUCTO"], as_index=False)["DEMANDA"].sum()
    costos_dict = costos_por_producto.set_index("PRODUCTO")["COSTO_UNITARIO"].to_dict()

    filas = []
    for _, fila in cantidades.iterrows():
        producto = fila["PRODUCTO"]
        costo = costos_dict.get(producto)
        precio = precios_por_producto.get(producto)
        if costo is None or precio is None:
            continue

        margen_unitario = precio - costo
        filas.append(
            {
                "REGIONAL": fila["REGIONAL"],
                "PRODUCTO": producto,
                "unique_id": f"{fila['REGIONAL']}|{producto}",
                "CANTIDAD_VENTANA": fila["DEMANDA"],
                "PRECIO": precio,
                "COSTO": costo,
                "MARGEN_UNITARIO": margen_unitario,
                "UTILIDAD_TOTAL": margen_unitario * fila["DEMANDA"],
            }
        )

    return pd.DataFrame(filas)


def clasificar_abc(
    df_utilidad: pd.DataFrame, corte_a: float = ABC_CORTE_A_DEFAULT, corte_b: float = ABC_CORTE_B_DEFAULT
) -> pd.DataFrame:
    """Clasificacion ABC por Pareto de utilidad acumulada: A hasta corte_a%,
    B hasta corte_b%, C el resto."""
    df = df_utilidad.sort_values("UTILIDAD_TOTAL", ascending=False).reset_index(drop=True)
    total = df["UTILIDAD_TOTAL"].sum()

    if total <= 0:
        df["PCT_ACUM_UTILIDAD"] = 0.0
        df["CLASE_ABC"] = "C"
        return df

    df["PCT_ACUM_UTILIDAD"] = df["UTILIDAD_TOTAL"].cumsum() / total * 100

    def _clase(pct):
        if pct <= corte_a:
            return "A"
        if pct <= corte_b:
            return "B"
        return "C"

    df["CLASE_ABC"] = df["PCT_ACUM_UTILIDAD"].apply(_clase)
    return df


def clasificar_xyz(df_scores: pd.DataFrame, corte_x: float, corte_y: float) -> pd.DataFrame:
    """Clasificacion XYZ por Score% del pronostico: X <= corte_x, Y <= corte_y, Z el resto."""

    def _clase(score_pct):
        if score_pct <= corte_x:
            return "X"
        if score_pct <= corte_y:
            return "Y"
        return "Z"

    df = df_scores.copy()
    df["CLASE_XYZ"] = df["SCORE_PORC"].apply(_clase)
    return df


def construir_pareto_abc(
    df_clasificacion: pd.DataFrame, corte_a: float = ABC_CORTE_A_DEFAULT, corte_b: float = ABC_CORTE_B_DEFAULT
):
    """Grafico de Pareto clasico: barras de utilidad por SKU (ordenadas de
    mayor a menor, color = clase ABC) con la linea de % acumulado, y lineas
    punteadas en los cortes A/B."""
    colores_abc = {"A": "#000C66", "B": "#00A9E0", "C": "#FF9900"}
    df_barras = df_clasificacion.sort_values("UTILIDAD_TOTAL", ascending=False)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df_barras["unique_id"],
            y=df_barras["UTILIDAD_TOTAL"],
            marker_color=[colores_abc.get(c, "gray") for c in df_barras["CLASE_ABC"]],
            name="Utilidad total",
            showlegend=False,
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_barras["unique_id"],
            y=df_barras["PCT_ACUM_UTILIDAD"],
            mode="lines+markers",
            name="% acumulado",
            line=dict(color="#1A1A1A"),
            showlegend=False,
        ),
        secondary_y=True,
    )

    for corte in (corte_a, corte_b):
        fig.add_hline(y=corte, line=dict(color="#57606A", dash="dot"), secondary_y=True)

    fig.update_layout(template="ggplot2", height=500, title="Pareto de utilidad por SKU (clasificacion ABC)")
    fig.update_xaxes(title_text="SKU")
    fig.update_yaxes(title_text="Utilidad total", secondary_y=False)
    fig.update_yaxes(title_text="% acumulado", range=[0, 105], secondary_y=True)
    return fig


def construir_matriz_abc_xyz(df_clasificacion: pd.DataFrame):
    """Matriz ABC-XYZ (columnas C-B-A por valor, filas X-Y-Z por variabilidad),
    coloreada por fila como en el material de clase: X=verde (baja
    variabilidad), Y=amarillo (media), Z=naranja/rojo (alta). Cada celda
    muestra los SKU que caen ahi."""
    columnas_abc = ["C", "B", "A"]
    filas_xyz = ["X", "Y", "Z"]

    def _linea_sku(fila_sku):
        return (
            f"{fila_sku['unique_id']}<br>"
            f"   Utilidad: ${fila_sku['UTILIDAD_TOTAL']:,.0f} · Score: {fila_sku['SCORE_PORC']:.1f}%"
        )

    texto = []
    hover = []
    for xyz in filas_xyz:
        fila_texto, fila_hover = [], []
        for abc in columnas_abc:
            sub = df_clasificacion[(df_clasificacion["CLASE_ABC"] == abc) & (df_clasificacion["CLASE_XYZ"] == xyz)]
            if sub.empty:
                fila_texto.append("—")
                fila_hover.append(f"Clase {abc}{xyz}: sin SKU")
            else:
                fila_texto.append("<br>".join(_linea_sku(r) for _, r in sub.iterrows()))
                fila_hover.append(
                    f"Clase {abc}{xyz} ({len(sub)} SKU)<br>" + "<br>".join(_linea_sku(r) for _, r in sub.iterrows())
                )
        texto.append(fila_texto)
        hover.append(fila_hover)

    z = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]  # una banda de color por fila (X, Y, Z)
    colorscale = [
        [0.0, "#8FBF8F"], [0.333, "#8FBF8F"],
        [0.333, "#F5C453"], [0.666, "#F5C453"],
        [0.666, "#E2703A"], [1.0, "#E2703A"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            x=columnas_abc,
            y=filas_xyz,
            z=z,
            zmin=1,
            zmax=3,
            colorscale=colorscale,
            text=texto,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#1A1A1A"),
            hovertext=hover,
            hoverinfo="text",
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )
    fig.update_layout(
        template="ggplot2",
        height=480,
        title="Matriz ABC-XYZ",
        xaxis=dict(title="Valor / utilidad (C -> B -> A)", side="bottom"),
        yaxis=dict(title="Variabilidad / pronosticabilidad (X -> Y -> Z)"),
    )
    return fig


def generar_conclusiones(df_clasificacion: pd.DataFrame) -> list:
    """Conclusiones en lenguaje de negocio, calculadas a partir de la
    clasificacion real (no son texto fijo)."""
    conclusiones = []
    total = df_clasificacion["UTILIDAD_TOTAL"].sum()
    n_sku = len(df_clasificacion)

    for clase in ["A", "B", "C"]:
        sub = df_clasificacion[df_clasificacion["CLASE_ABC"] == clase]
        if not sub.empty and total > 0:
            pct = sub["UTILIDAD_TOTAL"].sum() / total * 100
            conclusiones.append(
                f"La clase {clase} agrupa {len(sub)} de {n_sku} SKU y concentra el {pct:.1f}% de la utilidad total."
            )

    if not df_clasificacion.empty and total > 0:
        top = df_clasificacion.sort_values("UTILIDAD_TOTAL", ascending=False).iloc[0]
        pct_top = top["UTILIDAD_TOTAL"] / total * 100
        conclusiones.append(
            f"El SKU de mayor utilidad es {top['unique_id']}, con ${top['UTILIDAD_TOTAL']:,.0f} "
            f"({pct_top:.1f}% del total)."
        )

    riesgo = df_clasificacion[(df_clasificacion["CLASE_ABC"] == "A") & (df_clasificacion["CLASE_XYZ"] == "Z")]
    if not riesgo.empty:
        lista = ", ".join(riesgo["unique_id"])
        conclusiones.append(
            f"Atencion prioritaria: {lista} son de alto valor (clase A) pero alta variabilidad (clase Z); "
            "conviene reforzar el pronostico y el stock de seguridad de estos SKU."
        )
    else:
        conclusiones.append(
            "Ningun SKU de clase A cae en variabilidad Z: los productos de mayor valor son "
            "razonablemente predecibles con el modelo actual."
        )

    estables = df_clasificacion[(df_clasificacion["CLASE_ABC"] == "C") & (df_clasificacion["CLASE_XYZ"] == "X")]
    if not estables.empty:
        lista = ", ".join(estables["unique_id"])
        conclusiones.append(
            f"{lista} son de baja utilidad y baja variabilidad: se pueden manejar con tecnicas "
            "sencillas y revisiones poco frecuentes."
        )

    return conclusiones


def construir_pareto_abc_png(
    df_clasificacion: pd.DataFrame, corte_a: float = ABC_CORTE_A_DEFAULT, corte_b: float = ABC_CORTE_B_DEFAULT
) -> bytes:
    """Version del Pareto ABC dibujada con matplotlib (sin navegador), para
    incrustar en el PDF."""
    colores_abc = {"A": "#000C66", "B": "#00A9E0", "C": "#FF9900"}
    df = df_clasificacion.sort_values("UTILIDAD_TOTAL", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    ax1.bar(df["unique_id"], df["UTILIDAD_TOTAL"], color=[colores_abc.get(c, "gray") for c in df["CLASE_ABC"]])
    ax1.set_xlabel("SKU")
    ax1.set_ylabel("Utilidad total")
    ax1.tick_params(axis="x", rotation=35)
    for etiqueta in ax1.get_xticklabels():
        etiqueta.set_ha("right")

    ax2 = ax1.twinx()
    ax2.plot(df["unique_id"], df["PCT_ACUM_UTILIDAD"], color="#1A1A1A", marker="o")
    ax2.set_ylabel("% acumulado")
    ax2.set_ylim(0, 105)
    for corte in (corte_a, corte_b):
        ax2.axhline(corte, color="#57606A", linestyle="dotted")

    ax1.set_title("Pareto de utilidad por SKU (clasificacion ABC)")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def construir_matriz_abc_xyz_png(df_clasificacion: pd.DataFrame) -> bytes:
    """Version de la matriz ABC-XYZ dibujada con matplotlib (sin navegador),
    para incrustar en el PDF."""
    columnas_abc = ["C", "B", "A"]
    filas_xyz = ["X", "Y", "Z"]  # de abajo hacia arriba: Z queda arriba, X abajo
    colores_fila = {"X": "#8FBF8F", "Y": "#F5C453", "Z": "#E2703A"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, xyz in enumerate(filas_xyz):
        for j, abc in enumerate(columnas_abc):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colores_fila[xyz], edgecolor="white"))
            sub = df_clasificacion[
                (df_clasificacion["CLASE_ABC"] == abc) & (df_clasificacion["CLASE_XYZ"] == xyz)
            ]
            if sub.empty:
                texto = "—"
            else:
                texto = "\n".join(
                    f"{fila['unique_id']}\n${fila['UTILIDAD_TOTAL']:,.0f} · {fila['SCORE_PORC']:.1f}%"
                    for _, fila in sub.iterrows()
                )
            ax.text(j + 0.5, i + 0.5, texto, ha="center", va="center", fontsize=7, color="#1A1A1A")

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(columnas_abc)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(filas_xyz)
    ax.set_xlabel("Valor / utilidad (C -> B -> A)")
    ax.set_ylabel("Variabilidad / pronosticabilidad (X -> Y -> Z)")
    ax.set_title("Matriz ABC-XYZ")
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    return buffer.getvalue()


def construir_reporte_pdf_gerencial(df_clasificacion: pd.DataFrame) -> bytes:
    """Reporte de una a dos paginas para gerencia: KPIs, conclusiones en
    lenguaje de negocio y las dos graficas (Pareto ABC y matriz ABC-XYZ)."""
    NAVY = colors.HexColor("#000C66")
    GRIS = colors.HexColor("#EDEEF6")
    GRIS_TEXTO = colors.HexColor("#57606A")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=1.2 * cm, bottomMargin=1.2 * cm, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
    )
    ancho_util = doc.width

    styles = getSampleStyleSheet()
    estilo_titulo = ParagraphStyle("TituloReporte", parent=styles["Title"], textColor=colors.white, fontSize=22, leading=26, alignment=TA_CENTER)
    estilo_subtitulo = ParagraphStyle("Subtitulo", parent=styles["Normal"], textColor=colors.white, fontSize=11, alignment=TA_CENTER)
    estilo_h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=NAVY, spaceBefore=14, spaceAfter=6)
    estilo_body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14)
    estilo_bullet = ParagraphStyle("Bullet", parent=estilo_body, leftIndent=10, spaceAfter=6)
    estilo_kpi_valor = ParagraphStyle("KpiValor", fontSize=17, textColor=NAVY, alignment=TA_CENTER, leading=20)
    estilo_kpi_label = ParagraphStyle("KpiLabel", fontSize=9, textColor=GRIS_TEXTO, alignment=TA_CENTER)

    elementos = []

    encabezado = Table(
        [
            [Paragraph("ERP MotoTrack", estilo_titulo)],
            [Paragraph("Reporte gerencial &mdash; Clasificacion ABC / XYZ", estilo_subtitulo)],
            [Paragraph(pd.Timestamp.now().strftime("Generado el %d/%m/%Y"), estilo_subtitulo)],
        ],
        colWidths=[ancho_util],
    )
    encabezado.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("TOPPADDING", (0, 0), (-1, 0), 16),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 16),
        ("TOPPADDING", (0, 1), (-1, -1), 2),
    ]))
    elementos.append(encabezado)
    elementos.append(Spacer(1, 0.5 * cm))

    total_utilidad = df_clasificacion["UTILIDAD_TOTAL"].sum()
    kpis = [
        ("Utilidad total", f"${total_utilidad / 1_000_000:,.0f} MM"),
        ("SKU analizados", str(len(df_clasificacion))),
        ("SKU clase A", str((df_clasificacion["CLASE_ABC"] == "A").sum())),
        ("SKU clase Z", str((df_clasificacion["CLASE_XYZ"] == "Z").sum())),
    ]
    ancho_kpi = ancho_util / len(kpis)
    celdas_kpi = []
    for etiqueta, valor in kpis:
        celda = Table(
            [[Paragraph(f"<b>{valor}</b>", estilo_kpi_valor)], [Paragraph(etiqueta, estilo_kpi_label)]],
            colWidths=[ancho_kpi - 0.2 * cm],
        )
        celda.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), GRIS),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        celdas_kpi.append(celda)
    tabla_kpis = Table([celdas_kpi], colWidths=[ancho_kpi] * len(kpis))
    tabla_kpis.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    elementos.append(tabla_kpis)

    elementos.append(Paragraph("Conclusiones clave", estilo_h2))
    for conclusion in generar_conclusiones(df_clasificacion):
        elementos.append(Paragraph(f"&bull; {conclusion}", estilo_bullet))
    elementos.append(Paragraph(
        "Metodologia: ABC por utilidad acumulada (Pareto, cortes 80%/95%); XYZ por Score% del pronostico "
        "(X &le; 25%, 25% &lt; Y &le; 60%, Z &gt; 60%), siguiendo Dhoka &amp; Choudary (2013).",
        ParagraphStyle("Metodologia", parent=estilo_body, fontSize=8, textColor=GRIS_TEXTO, spaceBefore=4),
    ))

    elementos.append(Paragraph("Clasificacion ABC: Pareto de utilidad", estilo_h2))
    png_pareto = construir_pareto_abc_png(df_clasificacion)
    elementos.append(RLImage(io.BytesIO(png_pareto), width=ancho_util, height=ancho_util * 650 / 1000))

    elementos.append(Paragraph("Top 5 SKU por utilidad", estilo_h2))
    top5 = df_clasificacion.sort_values("UTILIDAD_TOTAL", ascending=False).head(5)
    filas_top5 = [["SKU", "Utilidad", "Clase ABC", "Score%", "Clase XYZ"]]
    for _, fila in top5.iterrows():
        filas_top5.append([
            fila["unique_id"], f"${fila['UTILIDAD_TOTAL']:,.0f}",
            fila["CLASE_ABC"], f"{fila['SCORE_PORC']:.1f}%", fila["CLASE_XYZ"],
        ])
    tabla_top5 = Table(filas_top5, colWidths=[ancho_util * w for w in (0.32, 0.28, 0.15, 0.13, 0.12)])
    tabla_top5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, GRIS]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D3D4D9")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elementos.append(tabla_top5)

    elementos.append(PageBreak())

    elementos.append(Paragraph("Matriz ABC-XYZ", estilo_h2))
    png_matriz = construir_matriz_abc_xyz_png(df_clasificacion)
    elementos.append(RLImage(io.BytesIO(png_matriz), width=ancho_util, height=ancho_util * 650 / 1000))
    elementos.append(Spacer(1, 0.3 * cm))
    elementos.append(Paragraph(
        "<b>Z</b> (alta variabilidad): validar productos nuevos y estacionalidad, usar tecnicas mas avanzadas "
        "(modelos predictivos, ML). <b>Y</b> (variabilidad media): revisiones mas frecuentes. "
        "<b>X</b> (baja variabilidad): tecnicas sencillas, revisiones menos frecuentes. "
        "<b>Columna C</b>: SKU poco importantes por utilidad, revisiones poco frecuentes sin importar su variabilidad.",
        estilo_body,
    ))

    doc.build(elementos)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 6. EOQ de materias primas
# ---------------------------------------------------------------------------

def _proveedor_por_material(nombre_material: str):
    """El proveedor de una materia prima se identifica por el color en su
    nombre (ej. 'LADRILLO COMPLETO AZUL' -> Proveedor Azul). Las materias
    GRIS no se usan en este ejercicio (no tienen proveedor asignado)."""
    for color in COLORES_A_PROVEEDOR:
        if nombre_material.endswith(color):
            return color
    return None


def extraer_materias_primas(archivo) -> pd.DataFrame:
    """Lee el catalogo de materias primas (nombre + volumen unitario) del
    bloque 'Raw Materials' en la hoja 'Products & Materials' del archivo de
    demanda. Excluye las materias GRIS."""
    archivo.seek(0)
    df_raw = pd.read_excel(archivo, sheet_name="Products & Materials", header=None)

    inicio = None
    for i in range(len(df_raw)):
        if str(df_raw.iloc[i, 0]).strip().lower() == "raw materials":
            inicio = i + 2  # +1 = fila de encabezados (Name, Width, Depth, Height...), +2 = primer dato
            break
    if inicio is None:
        raise ValueError("No se encontro el bloque 'Raw Materials' en la hoja 'Products & Materials'.")

    filas = []
    for i in range(inicio, len(df_raw)):
        nombre = df_raw.iloc[i, 0]
        if pd.isna(nombre):
            break
        ancho, profundo, alto = df_raw.iloc[i, 1], df_raw.iloc[i, 2], df_raw.iloc[i, 3]
        filas.append(
            {
                "MATERIA_PRIMA": str(nombre).strip().upper(),
                "VOLUMEN_UNITARIO": float(ancho) * float(profundo) * float(alto),
            }
        )

    df = pd.DataFrame(filas)
    return df[df["MATERIA_PRIMA"].apply(lambda n: _proveedor_por_material(n) is not None)].reset_index(drop=True)


def extraer_bom(archivo) -> pd.DataFrame:
    """Lee la relacion Producto -> Materia Prima (cuantas unidades de cada
    insumo lleva cada producto) del bloque 'Product - Raw Material
    Relationship' en 'Products & Materials'. Formato largo:
    PRODUCTO | MATERIA_PRIMA | CANTIDAD_POR_UNIDAD."""
    archivo.seek(0)
    df_raw = pd.read_excel(archivo, sheet_name="Products & Materials", header=None)

    inicio = None
    for i in range(len(df_raw)):
        if str(df_raw.iloc[i, 0]).strip().lower() == "product - raw material relationship":
            inicio = i + 1  # fila de encabezados: "Product \ Raw Material", <materias primas...>
            break
    if inicio is None:
        raise ValueError("No se encontro el bloque 'Product - Raw Material Relationship'.")

    encabezados = df_raw.iloc[inicio]
    materiales = [str(v).strip().upper() if pd.notna(v) else None for v in encabezados]

    filas = []
    for i in range(inicio + 1, len(df_raw)):
        producto = df_raw.iloc[i, 0]
        if pd.isna(producto):
            break
        for j in range(1, len(materiales)):
            material = materiales[j]
            if material is None:
                continue
            cantidad = df_raw.iloc[i, j]
            if pd.notna(cantidad) and cantidad > 0:
                filas.append(
                    {
                        "PRODUCTO": str(producto).strip().upper(),
                        "MATERIA_PRIMA": material,
                        "CANTIDAD_POR_UNIDAD": float(cantidad),
                    }
                )

    return pd.DataFrame(filas)


def extraer_transporte_proveedores(archivo) -> pd.DataFrame:
    """Lee la hoja 'Transportation' y se queda con las rutas Proveedor ->
    Fabrica: modo de transporte, precio fijo por viaje y capacidad (u3)."""
    archivo.seek(0)
    df = pd.read_excel(archivo, sheet_name="Transportation")
    df = df[df["Origin"].astype(str).str.contains("Proveedor", case=False, na=False)].copy()

    df["PROVEEDOR"] = df["Origin"].astype(str).str.extract(r"[Pp]roveedor\s+(\w+)", expand=False).str.upper()
    df["FIXED_PRICE"] = pd.to_numeric(df["Fixed Price Per Turn"], errors="coerce")
    df["CAPACIDAD_U3"] = pd.to_numeric(df["Capacity u3"], errors="coerce")

    return df.rename(columns={"Transportation Mean": "MODO", "Turns To Destination": "TURNOS_ENTREGA"})[
        ["PROVEEDOR", "MODO", "FIXED_PRICE", "CAPACIDAD_U3", "TURNOS_ENTREGA"]
    ].reset_index(drop=True)


def extraer_precio_materias_primas(ruta_o_archivo) -> pd.DataFrame:
    """Busca en la hoja 'Costos' el bloque 'Materia Prima | Precio Unitario
    (COP)' (seccion 1) y devuelve MATERIA_PRIMA -> COSTO_UNITARIO."""
    df_raw = pd.read_excel(ruta_o_archivo, sheet_name="Costos", header=None)

    fila_header = None
    for i in range(len(df_raw)):
        if str(df_raw.iloc[i, 0]).strip().lower() == "materia prima":
            fila_header = i
            break
    if fila_header is None:
        raise ValueError("No se encontro en la hoja 'Costos' el bloque 'Materia Prima / Precio Unitario'.")

    filas = []
    for i in range(fila_header + 1, len(df_raw)):
        nombre = df_raw.iloc[i, 0]
        precio = df_raw.iloc[i, 1]
        if pd.isna(nombre) or pd.isna(precio):
            break
        filas.append({"MATERIA_PRIMA": str(nombre).strip().upper(), "COSTO_UNITARIO": float(precio)})

    if not filas:
        raise ValueError("El bloque 'Materia Prima / Precio Unitario' no tiene filas de datos.")

    return pd.DataFrame(filas)


def calcular_demanda_anual_materia_prima(df_long_completo: pd.DataFrame, bom: pd.DataFrame) -> pd.DataFrame:
    """Demanda anual (ultimas 52 semanas) de motos a nivel MOTOTRAK (las 3
    sedes), traducida a consumo anual de materia prima via el BOM."""
    ventana = aplicar_ventana_movil(df_long_completo, semanas=VENTANA_SEMANAS)
    demanda_anual_producto = (
        ventana[ventana["REGIONAL"] == "MOTOTRAK"]
        .groupby("PRODUCTO", as_index=False)["DEMANDA"]
        .sum()
        .rename(columns={"DEMANDA": "DEMANDA_ANUAL"})
    )

    combinado = bom.merge(demanda_anual_producto, on="PRODUCTO", how="inner")
    combinado["CONSUMO_ANUAL"] = combinado["CANTIDAD_POR_UNIDAD"] * combinado["DEMANDA_ANUAL"]

    return combinado.groupby("MATERIA_PRIMA", as_index=False)["CONSUMO_ANUAL"].sum()


def calcular_eoq_materias_primas(
    demanda_anual: pd.DataFrame,
    precios_mp: pd.DataFrame,
    materias_primas: pd.DataFrame,
    transporte: pd.DataFrame,
    tasa_inventario_pct: float,
) -> pd.DataFrame:
    """EOQ por materia prima (EOQ = raiz(2*D*S/H)). Para el proveedor Negro,
    que tiene dos medios de transporte (Ship y Plane), genera una fila por
    cada uno para poder comparar el costo logistico total."""
    base = demanda_anual.merge(precios_mp, on="MATERIA_PRIMA", how="inner")
    base = base.merge(materias_primas, on="MATERIA_PRIMA", how="inner")
    base["PROVEEDOR"] = base["MATERIA_PRIMA"].apply(_proveedor_por_material)
    base = base[base["PROVEEDOR"].notna()]

    filas = []
    for _, fila in base.iterrows():
        rutas = transporte[transporte["PROVEEDOR"] == fila["PROVEEDOR"]]
        for _, ruta in rutas.iterrows():
            D = fila["CONSUMO_ANUAL"]
            S = ruta["FIXED_PRICE"]
            H = tasa_inventario_pct / 100 * fila["COSTO_UNITARIO"]
            if D <= 0 or H <= 0 or pd.isna(S):
                continue

            eoq = np.sqrt(2 * D * S / H)
            volumen_pedido = eoq * fila["VOLUMEN_UNITARIO"]
            viajes = int(np.ceil(volumen_pedido / ruta["CAPACIDAD_U3"])) if ruta["CAPACIDAD_U3"] > 0 else 1
            s_efectivo = S * viajes
            pedidos_por_anio = D / eoq
            costo_ordenar_anual = pedidos_por_anio * s_efectivo
            costo_mantener_anual = (eoq / 2) * H

            filas.append(
                {
                    "MATERIA_PRIMA": fila["MATERIA_PRIMA"],
                    "PROVEEDOR": fila["PROVEEDOR"],
                    "MODO_TRANSPORTE": ruta["MODO"],
                    "DEMANDA_ANUAL": round(D, 1),
                    "COSTO_UNITARIO": fila["COSTO_UNITARIO"],
                    "S_POR_VIAJE": S,
                    "H_ANUAL": round(H, 1),
                    "EOQ": round(eoq, 1),
                    "VIAJES_POR_PEDIDO": viajes,
                    "S_EFECTIVO": s_efectivo,
                    "PEDIDOS_POR_ANIO": round(pedidos_por_anio, 2),
                    "COSTO_ORDENAR_ANUAL": round(costo_ordenar_anual, 0),
                    "COSTO_MANTENER_ANUAL": round(costo_mantener_anual, 0),
                    "COSTO_TOTAL_ANUAL": round(costo_ordenar_anual + costo_mantener_anual, 0),
                }
            )

    return pd.DataFrame(filas)


def construir_grafica_eoq(df_eoq: pd.DataFrame):
    """Barras de EOQ por materia prima (color = proveedor) y, para el
    proveedor Negro, comparacion del costo logistico total anual entre
    Ship y Plane."""
    colores_proveedor = {"AZUL": "royalblue", "AMARILLO": "goldenrod", "NEGRO": "black"}

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["EOQ por materia prima (color = proveedor)", "Proveedor Negro: costo total anual, Ship vs Plane"],
    )

    etiquetas = df_eoq["MATERIA_PRIMA"] + " (" + df_eoq["MODO_TRANSPORTE"].astype(str) + ")"
    fig.add_trace(
        go.Bar(
            x=etiquetas,
            y=df_eoq["EOQ"],
            marker_color=[colores_proveedor.get(p, "gray") for p in df_eoq["PROVEEDOR"]],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    negro = df_eoq[df_eoq["PROVEEDOR"] == "NEGRO"]
    colores_modo = {"ship": "steelblue", "plane": "salmon"}
    for modo, color in colores_modo.items():
        sub = negro[negro["MODO_TRANSPORTE"].astype(str).str.lower() == modo]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(x=sub["MATERIA_PRIMA"], y=sub["COSTO_TOTAL_ANUAL"], name=modo.capitalize(), marker_color=color),
            row=1,
            col=2,
        )

    fig.update_layout(template="ggplot2", height=550, barmode="group")
    fig.update_xaxes(title_text="Materia prima", row=1, col=1)
    fig.update_yaxes(title_text="EOQ (unidades)", row=1, col=1)
    fig.update_xaxes(title_text="Materia prima", row=1, col=2)
    fig.update_yaxes(title_text="Costo total anual (COP)", row=1, col=2)
    return fig


# ---------------------------------------------------------------------------
# Interfaz de Streamlit
# ---------------------------------------------------------------------------

st.html("""
<style>
.st-key-header_banner {
    background-color: #000C66;
    padding: 1.25rem 1.5rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.st-key-header_banner * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
}
.st-key-tabs_principales [role="tablist"] {
    background-color: transparent;
    border-bottom: none;
    gap: 0.75rem;
    padding: 0;
}
.st-key-tabs_principales [role="tab"] {
    color: #000C66 !important;
    background-color: #EDEEF6;
    border-radius: 999px;
    padding: 0.55rem 1.25rem;
}
.st-key-tabs_principales [role="tab"] svg {
    fill: #000C66 !important;
}
.st-key-tabs_principales [role="tab"][aria-selected="true"] {
    background-color: #000C66;
    color: #FFFFFF !important;
}
.st-key-tabs_principales [role="tab"][aria-selected="true"] svg {
    fill: #FFFFFF !important;
}
.st-key-tabs_principales .react-aria-SelectionIndicator {
    display: none;
}
</style>
""")
with st.container(key="header_banner"):
    st.title("ERP MotoTrack", icon=":material/factory:")

if MOSTRAR_TAB_EOQ:
    tab_pronostico, tab_clasificacion, tab_eoq = st.tabs(
        [":material/trending_up: Pronostico de demanda",
         ":material/inventory_2: Clasificacion ABC y XYZ",
         ":material/local_shipping: EOQ de materias primas"],
        key="tabs_principales",
    )
else:
    tab_pronostico, tab_clasificacion = st.tabs(
        [":material/trending_up: Pronostico de demanda",
         ":material/inventory_2: Clasificacion ABC y XYZ"],
        key="tabs_principales",
    )

# ---- Pestana 1: Pronostico de demanda -------------------------------------
with tab_pronostico:
    st.header("Motor de Pronosticos MotoTrack", icon=":material/trending_up:")
    st.write(
        "Carga el historico de demanda, compara modelos mediante backtesting "
        "y obten el pronostico del mejor modelo para cada serie."
    )

    archivo = st.file_uploader(
        "Sube el archivo moto-track.xlsx", type=["xlsx"], key="uploader_demanda"
    )

    col1, col2 = st.columns(2)
    with col1:
        n_windows = st.number_input(
            "n_windows (periodos hacia atras para backtesting)", min_value=1, value=6, step=1
        )
    with col2:
        h = st.number_input("h (periodos a pronosticar)", min_value=1, value=6, step=1)

    if archivo is not None:
        try:
            df_long_crudo = cargar_datos(archivo)
            # Guardamos el historico completo (para la utilidad de 4 anios en
            # la pestana de clasificacion) y, aparte, la ventana de 52 semanas
            # que usa el motor de pronostico.
            st.session_state["df_long_completo"] = df_long_crudo
            st.session_state["df_long"] = aplicar_ventana_movil(df_long_crudo)
        except Exception as e:
            # Si el archivo nuevo es invalido, no dejamos datos viejos a medias:
            # el usuario debe subir uno valido antes de poder generar el pronostico.
            st.session_state.pop("df_long", None)
            st.session_state.pop("df_long_completo", None)
            st.error(f"No se pudo leer el archivo. Verifica el formato de la hoja 'Demand'. Detalle: {e}")
        else:
            # El mismo archivo trae ademas la lista de materiales y las rutas
            # de transporte que usa la pestana de EOQ. Si esto falla, el
            # pronostico de demanda sigue funcionando igual; solo se avisa
            # que el EOQ no va a estar disponible.
            try:
                st.session_state["bom"] = extraer_bom(archivo)
                st.session_state["materias_primas"] = extraer_materias_primas(archivo)
                st.session_state["transporte"] = extraer_transporte_proveedores(archivo)
            except Exception as e:
                st.session_state.pop("bom", None)
                st.session_state.pop("materias_primas", None)
                st.session_state.pop("transporte", None)
                st.warning(
                    "El pronostico de demanda quedo listo, pero no se pudo leer la "
                    "informacion de materiales/transporte para la pestana de EOQ. "
                    f"Detalle: {e}"
                )

    generar = st.button("Generar pronostico", icon=":material/play_arrow:", type="primary")

    if generar:
        if "df_long" not in st.session_state:
            st.error("Primero sube un archivo Excel valido.")
        else:
            with st.spinner("Corriendo backtesting y generando pronosticos, esto puede tardar unos minutos..."):
                df_long = st.session_state["df_long"]
                df_ts = a_formato_statsforecast(df_long)

                df_cv, errores_cv, modelos = ejecutar_backtesting(df_ts, int(n_windows), int(h))

            if df_cv.empty:
                st.error(
                    "Ningun modelo pudo entrenarse con este archivo: probablemente el historico de "
                    "demanda es muy corto para los parametros actuales (n_windows="
                    f"{int(n_windows)}, h={int(h)}; algunos modelos tambien necesitan al menos 13 "
                    "semanas por su estacionalidad). Sube un archivo con mas turnos, o reduce "
                    "n_windows/h, y vuelve a intentar."
                )
            else:
                df_metricas = calcular_metricas(df_cv)
                ganadores = elegir_mejor_modelo(df_metricas)
                df_forecast, errores_fcst = generar_pronostico_final(df_ts, ganadores, modelos, int(h))
                tabla_resumen = construir_tabla_resumen(ganadores, df_forecast)
                fig = construir_grafica(df_long, df_forecast, ganadores, int(h))

                st.session_state["tabla_resumen"] = tabla_resumen
                st.session_state["fig"] = fig
                st.session_state["errores"] = errores_cv + errores_fcst

    if "tabla_resumen" in st.session_state:
        st.subheader("Tabla resumen", icon=":material/table_chart:")
        st.dataframe(st.session_state["tabla_resumen"], width="stretch")

        excel_bytes = df_a_excel_bytes(st.session_state["tabla_resumen"])
        st.download_button(
            "Descargar resumen en Excel",
            data=excel_bytes,
            file_name="resumen_pronostico_mototrack.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/download:",
        )

        st.subheader("Grafica", icon=":material/bar_chart:")
        st.plotly_chart(st.session_state["fig"], width="stretch")

        if st.session_state.get("errores"):
            with st.expander(f"Avisos durante el calculo ({len(st.session_state['errores'])})"):
                for err in st.session_state["errores"]:
                    st.write(f"- {err}")

# ---- Pestana 2: Clasificacion ABC y XYZ ------------------------------------
with tab_clasificacion:
    st.header("Clasificacion ABC y XYZ MotoTrack", icon=":material/inventory_2:")
    st.write(
        "Carga los costos por producto (hoja 'Costos'), ajusta el precio de "
        "venta y obten la clasificacion ABC (por utilidad) y XYZ (por Score% "
        "del pronostico) de cada SKU."
    )

    archivo_costos = st.file_uploader(
        "Sube el archivo de costos y precios (hoja 'Costos')", type=["xlsx"], key="uploader_costos"
    )

    if archivo_costos is not None:
        try:
            # Validamos que se pueda leer antes de sobreescribir el archivo
            # guardado, para no perder un costeo valido por una subida mala.
            extraer_costo_por_producto(archivo_costos)
            archivo_costos.seek(0)
            guardar_costos(archivo_costos)
            st.success("Archivo de costos actualizado y guardado para futuras sesiones.")
        except Exception as e:
            st.error(
                "No se pudo leer el bloque de costo unitario del archivo subido; se "
                f"conserva el archivo de costos anterior (si existia). Detalle: {e}"
            )

    st.subheader("Precio de venta por producto")
    precios_guardados = cargar_precios()
    col_p1, col_p2, col_p3 = st.columns(3)
    precios_actuales = {}
    for col, producto in zip((col_p1, col_p2, col_p3), ["MOTO", "CUATRIMOTO", "TRACTOR"]):
        with col:
            precios_actuales[producto] = st.number_input(
                f"Precio {producto} (COP)",
                min_value=0,
                value=int(precios_guardados[producto]),
                step=100_000,
                key=f"precio_{producto}",
            )
    guardar_precios(precios_actuales)

    h_clasificacion = st.number_input(
        "h (periodos a clasificar)", min_value=1, value=VENTANA_UTILIDAD_DEFAULT, step=1, key="h_clasificacion"
    )

    st.subheader("Umbrales XYZ (Score%)")
    st.caption(
        "Por defecto X <= 25%, 25% < Y <= 60%, Z > 60%, segun los limites de Score% "
        "del curso (Dhoka & Choudary, 2013). Ajustalos si tus datos lo requieren."
    )
    col_x, col_y = st.columns(2)
    with col_x:
        umbral_x = st.number_input(
            "X: Score% menor o igual a", min_value=0.0, value=XYZ_CORTE_X_DEFAULT, step=1.0, key="umbral_xyz_x"
        )
    with col_y:
        umbral_y = st.number_input(
            "Y: Score% menor o igual a", min_value=0.0, value=XYZ_CORTE_Y_DEFAULT, step=1.0, key="umbral_xyz_y"
        )

    generar_clasificacion = st.button("Generar clasificacion ABC o XYZ", icon=":material/play_arrow:", type="primary")

    if generar_clasificacion:
        faltantes = []
        if "df_long_completo" not in st.session_state:
            faltantes.append("subir el archivo de demanda en la pestana 'Pronostico de demanda'")
        if "tabla_resumen" not in st.session_state:
            faltantes.append("generar el pronostico en la pestana 'Pronostico de demanda' (se necesita el Score% y RMSE)")
        if not COSTOS_PATH.exists():
            faltantes.append("subir un archivo de costos valido")

        if faltantes:
            st.error("Antes de clasificar, falta: " + "; ".join(faltantes) + ".")
        else:
            try:
                df_costos = cargar_costos_guardados()
            except Exception as e:
                df_costos = None
                st.error(f"No se pudo leer el archivo de costos guardado. Detalle: {e}")

            if df_costos is not None:
                df_utilidad = calcular_utilidad_por_sku(
                    st.session_state["df_long_completo"], df_costos, precios_actuales, int(h_clasificacion)
                )
                df_abc = clasificar_abc(df_utilidad)

                df_scores = st.session_state["tabla_resumen"][
                    st.session_state["tabla_resumen"]["REGIONAL"].isin(REGIONALES_SKU)
                ][["REGIONAL", "PRODUCTO", "SCORE_PORC", "RMSE"]].copy()
                df_scores["unique_id"] = df_scores["REGIONAL"] + "|" + df_scores["PRODUCTO"]
                df_xyz = clasificar_xyz(df_scores, umbral_x, umbral_y)

                df_clasificacion = df_abc.merge(
                    df_xyz[["unique_id", "SCORE_PORC", "RMSE", "CLASE_XYZ"]], on="unique_id", how="left"
                )
                st.session_state["df_clasificacion"] = df_clasificacion

    if "df_clasificacion" in st.session_state:
        df_clasificacion = st.session_state["df_clasificacion"]

        st.subheader("Tabla de clasificacion", icon=":material/table_chart:")
        columnas_mostrar = [
            "REGIONAL", "PRODUCTO", "CANTIDAD_VENTANA", "PRECIO", "COSTO",
            "MARGEN_UNITARIO", "UTILIDAD_TOTAL", "PCT_ACUM_UTILIDAD", "CLASE_ABC",
            "SCORE_PORC", "RMSE", "CLASE_XYZ",
        ]
        st.dataframe(df_clasificacion[columnas_mostrar], width="stretch")

        excel_bytes_clasificacion = df_a_excel_bytes(df_clasificacion[columnas_mostrar])
        st.download_button(
            "Descargar clasificacion en Excel",
            data=excel_bytes_clasificacion,
            file_name="clasificacion_abc_xyz_mototrack.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/download:",
        )

        if df_clasificacion["CLASE_XYZ"].isna().any():
            sin_score = df_clasificacion.loc[df_clasificacion["CLASE_XYZ"].isna(), "unique_id"].tolist()
            st.warning(
                "Estos SKU no tienen Score% en la tabla de pronostico (vuelve a "
                f"correr 'Generar pronostico'): {', '.join(sin_score)}"
            )

        df_valida = df_clasificacion.dropna(subset=["CLASE_XYZ"])
        fig_pareto = construir_pareto_abc(df_valida)
        fig_matriz = construir_matriz_abc_xyz(df_valida)

        st.subheader("Pareto ABC", icon=":material/bar_chart:")
        st.plotly_chart(fig_pareto, width="stretch")

        st.subheader("Matriz ABC-XYZ", icon=":material/grid_view:")
        st.plotly_chart(fig_matriz, width="stretch")
        st.caption(
            "**Z** (alta variabilidad): validar productos nuevos y estacionalidad; usar tecnicas mas avanzadas "
            "(modelos predictivos, ML).  \n"
            "**Y** (variabilidad media): validar estacionalidad, tecnicas mas sofisticadas, revisiones mas frecuentes.  \n"
            "**X** (baja variabilidad): tecnicas sencillas, revisiones menos frecuentes.  \n"
            "**Columna C**: SKU poco importantes por utilidad, revisiones poco frecuentes sin importar su variabilidad."
        )

        st.subheader("Reporte para gerencia", icon=":material/picture_as_pdf:")
        try:
            pdf_bytes = construir_reporte_pdf_gerencial(df_valida)
            st.download_button(
                "Descargar reporte en PDF",
                data=pdf_bytes,
                file_name="reporte_gerencial_abc_xyz_mototrack.pdf",
                mime="application/pdf",
                icon=":material/download:",
                type="primary",
            )
        except Exception as e:
            st.error(f"No se pudo generar el reporte PDF. Detalle: {e}")

# ---- Pestana 3: EOQ de materias primas (oculta por ahora, ver MOSTRAR_TAB_EOQ) --
if MOSTRAR_TAB_EOQ:
    with tab_eoq:
        st.header("EOQ de Materias Primas MotoTrack", icon=":material/local_shipping:")
        st.write(
            "Calcula la cantidad economica de pedido (EOQ) de cada materia prima, "
            "a partir de la demanda anual de MOTOTRAK (traducida via la lista de "
            "materiales), el costo de mantener inventario y el costo de "
            "transporte de cada proveedor."
        )

        tasa_inventario = st.number_input(
            "Tasa de inventario anual (% EA)",
            min_value=0.0, value=TASA_INVENTARIO_DEFAULT, step=0.5, key="tasa_inventario",
        )

        generar_eoq = st.button("Generar EOQ", icon=":material/play_arrow:", type="primary")

        if generar_eoq:
            faltantes = []
            if "df_long_completo" not in st.session_state:
                faltantes.append("subir el archivo de demanda en la pestana 'Pronostico de demanda'")
            if not all(k in st.session_state for k in ("bom", "materias_primas", "transporte")):
                faltantes.append(
                    "volver a subir el archivo de demanda (no se pudo leer 'Products & Materials' / 'Transportation')"
                )
            if not COSTOS_PATH.exists():
                faltantes.append("subir un archivo de costos valido en la pestana de clasificacion")

            if faltantes:
                st.error("Antes de calcular el EOQ, falta: " + "; ".join(faltantes) + ".")
            else:
                try:
                    precios_mp = extraer_precio_materias_primas(COSTOS_PATH)
                except Exception as e:
                    precios_mp = None
                    st.error(f"No se pudo leer los precios de materia prima de la hoja 'Costos'. Detalle: {e}")

                if precios_mp is not None:
                    demanda_anual = calcular_demanda_anual_materia_prima(
                        st.session_state["df_long_completo"], st.session_state["bom"]
                    )
                    df_eoq = calcular_eoq_materias_primas(
                        demanda_anual,
                        precios_mp,
                        st.session_state["materias_primas"],
                        st.session_state["transporte"],
                        tasa_inventario,
                    )
                    st.session_state["df_eoq"] = df_eoq

        if "df_eoq" in st.session_state:
            df_eoq = st.session_state["df_eoq"]

            st.subheader("Tabla EOQ", icon=":material/table_chart:")
            st.dataframe(df_eoq, width="stretch")

            excel_bytes_eoq = df_a_excel_bytes(df_eoq)
            st.download_button(
                "Descargar EOQ en Excel",
                data=excel_bytes_eoq,
                file_name="eoq_materias_primas_mototrack.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/download:",
            )

            st.subheader("Grafica", icon=":material/bar_chart:")
            st.plotly_chart(construir_grafica_eoq(df_eoq), width="stretch")
