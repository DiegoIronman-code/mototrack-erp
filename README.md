# Motor de Pronosticos MotoTrack

App en Streamlit que carga el historico de demanda de MotoTrack (`moto-track.xlsx`,
hoja `Demand`), compara varios modelos de pronostico mediante backtesting y
muestra cual es el mejor modelo para cada serie (REGIONAL + PRODUCTO), con su
pronostico y sus graficas.

## Forma mas facil de abrir la app

Haz doble clic en **`iniciar_app.bat`**. La primera vez preparara el entorno
virtual e instalara las dependencias (puede tardar unos minutos); las
siguientes veces solo activara el entorno y levantara la app. Si ocurre un
error, la ventana se queda abierta para poder leer el mensaje.

## Pasos manuales (opcional)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Uso

1. Sube el archivo `moto-track.xlsx` en el uploader.
2. Ajusta `n_windows` (periodos hacia atras para el backtesting, por defecto 6)
   y `h` (periodos a pronosticar, por defecto 6) si lo necesitas.
3. Presiona **"Generar pronostico"**.
4. Revisa la tabla resumen y la grafica, y descarga el resumen en Excel si lo
   necesitas.

El archivo se puede volver a subir en cualquier momento con datos
actualizados de un nuevo turno; al presionar de nuevo el boton se recalcula
todo.
