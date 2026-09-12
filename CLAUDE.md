# Ejercicio: Motor de Pronósticos MotoTrack

## Objetivo

Construir una app en **Streamlit** que cargue el histórico de demanda de MotoTrack,
compare varios modelos de pronóstico mediante **backtesting** y muestre cuál es el
mejor para cada serie, con su pronóstico y sus gráficas.

**Entregable:** un solo archivo `app.py` + `requirements.txt`, que corra localmente.

---

## 1. Datos de entrada

Archivo Excel `moto-track.xlsx`, hoja **`Demand`**. Cuidado con el formato, no es una tabla plana:

- **Fila 1:** nombres decorativos de las tiendas (`MotoTrack Norte`, `Centro`, `Sur`) → ignorar.
- **Fila 2:** los encabezados reales → en pandas usar `header=1`.
- **Fila 3 en adelante:** los datos.
- Hay **tres bloques horizontales**: Norte (columnas A–D), Centro (F–I), Sur (K–M),
  separados por columnas vacías (E y J).
- Cada bloque tiene: `Turn`, `MOTO`, `CUATRIMOTO`, `TRACTOR`.
- **Ojo:** el bloque Sur **no tiene TRACTOR**. No asumas 3 productos por regional:
  lee las columnas que existan en cada bloque y descarta los valores nulos.
- `Turn` es un entero consecutivo (1, 2, 3...). Un turno = un periodo.

Transforma los datos a **formato largo**: `Turn | REGIONAL | PRODUCTO | DEMANDA`.

Después agrega dos regionales **calculadas** (sumando la demanda):

- `CEDI` = CENTRO + SUR
- `MOTOTRAK` = NORTE + CENTRO + SUR

Cada combinación **(REGIONAL, PRODUCTO)** es una serie independiente a pronosticar.

---

## 2. Funcionalidad de la app

1. Un `st.file_uploader` para el archivo `.xlsx`. Debe poder **volver a subirse cada turno**
   con datos actualizados y recalcular todo.
2. Dos parámetros editables por el usuario:
   - `n_windows`: periodos hacia atrás para el backtesting → **valor por defecto 6**
   - `h`: cantidad de periodos a pronosticar → **valor por defecto 6**
3. Un botón "Generar pronóstico".
4. Mostrar: tabla resumen + gráfica + botón para descargar el resumen en Excel.

---

## 3. Modelos a evaluar (librería `statsforecast` de Nixtla)

```python
from statsforecast.models import (
    HoltWinters,
    WindowAverage,
    SimpleExponentialSmoothingOptimized,
    MSTL,
)

modelos = [
    HoltWinters(season_length=13, alias='hw'),
    WindowAverage(window_size=3,  alias='wa_3'),
    WindowAverage(window_size=6,  alias='wa_6'),
    WindowAverage(window_size=12, alias='wa_12'),
    SimpleExponentialSmoothingOptimized(alias='ses'),
    MSTL(season_length=13, alias='mstl'),
]
```

`StatsForecast` espera un DataFrame con las columnas `unique_id`, `ds`, `y`:

- `unique_id` = `"REGIONAL|PRODUCTO"`
- `ds` = convierte el `Turn` a fechas ficticias semanales (`freq="W-MON"`)
- `y` = la demanda

Si un modelo falla en alguna serie, captura el error y sigue con los demás.

---

## 4. Backtesting y métricas

Usar la validación cruzada de la librería:

```python
cv = sf.cross_validation(df=df_ts, h=h, n_windows=n_windows, step_size=h)
```

Sobre esos resultados, calcular para **cada serie y cada modelo**:

| Métrica  | Fórmula |
|----------|---------|
| MAE%     | `Σ|y − pred| / Σy` |
| Sesgo%   | `Σ(y − pred) / Σy` |
| **Score%** | **`MAE% + |Sesgo%|`** |
| RMSE     | `sqrt(mean((y − pred)²))` |

El **mejor modelo de cada serie es el de menor Score%**. Con ese modelo se genera
el pronóstico final de `h` periodos hacia adelante.

---

## 5. Tabla resumen

Una fila por serie, con las columnas:

`REGIONAL` | `PRODUCTO` | `MODELO` | `SCORE_PORC` (en %, 1 decimal) | `RMSE` | y una columna por cada turno pronosticado

Agregar un botón para descargarla en Excel.

---

## 6. Gráfica (Plotly)

Una figura con **subplots de 2 filas x 3 columnas**, con estos títulos:

`"NORTE"`, `"CENTRO"`, `"SUR"`, `"CEDI (C+S)"`, `"MOTOTRAK (N+C+S)"`, `""` (la sexta va vacía)

Dentro de cada subplot, para cada producto:

- **Línea sólida** = demanda real (mostrar solo los últimos `52 + h` turnos)
- **Línea punteada** (`dash='dot'`) = pronóstico del mejor modelo, del mismo color
- Colores: `MOTO` = `salmon`, `CUATRIMOTO` = `navy`, `TRACTOR` = `darkcyan`

Detalles: leyenda visible una sola vez, `template="ggplot2"`, `height=700`,
eje X con título `"Turn"`, eje Y con título `"Demanda"`.

---

## 7. Ejecución local

Generar el `requirements.txt` (streamlit, pandas, numpy, plotly, statsforecast, openpyxl)
y dejar en un `README.md` los pasos para correrla:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 8. Restricciones del ejercicio

Es un ejercicio para principiantes, así que:

- Todo en un solo archivo `app.py`, con **comentarios en español**.
- Sin base de datos, sin login, sin despliegue en la nube.
- Usar `st.session_state` para no perder los resultados al interactuar con la app.
- Si algo falla al leer el Excel, mostrar un mensaje claro con `st.error`, no un traceback.

---

## 9. Acceso directo con un clic (archivo .bat)

Además de la app, crea en la carpeta de trabajo un archivo **`iniciar_app.bat`** que
permita abrir el aplicativo **con solo hacer doble clic**, sin escribir comandos.

El .bat debe:

- Ubicarse **al lado de `app.py`** y funcionar sin importar desde dónde se ejecute
  (usar `cd /d "%~dp0"`).
- La **primera vez**: crear el entorno virtual e instalar las dependencias.
- Las **siguientes veces**: solo activar el entorno y levantar la app.
- Dejar la ventana abierta si ocurre un error, para poder leer el mensaje.

Contenido sugerido:

```bat
@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Preparando el entorno por primera vez, esto puede tardar unos minutos...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate.bat
)

echo Iniciando la aplicacion...
streamlit run app.py

pause
```

Menciona este archivo en el `README.md` como la forma más fácil de abrir la app.
