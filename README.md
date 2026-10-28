# 🇨🇴 SpVAR Colombia – Educación y Crecimiento Económico

Aplicación interactiva con **Gradio** que implementa un modelo de **Vector Autorregresivo Espacial (SpVAR)** y Análisis Exploratorio de Datos Espaciales (ESDA) para los 33 departamentos de Colombia.

## Características

- Mapas coropléticos de alto espectro
- Mapa interactivo Folium (base oscura CartoDB)
- Visualizaciones 3D (Scatter, Superficie, Animación temporal)
- Estimación de los **4 mecanismos de transmisión espacial**
- Impactos directos / indirectos / totales (LeSage & Pace)
- Informe ejecutivo descargable

## Fuentes 100% públicas

| Recurso | Origen |
|---------|--------|
| Geometría de departamentos | [GitHub raw – caticoa3/colombia_mapa](https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson) |
| Tiles del mapa | CartoDB dark_matter (público) |
| Datos de análisis | Sintéticos generados localmente |
| Gradio share | Servicio público de Gradio |

**No se requieren API keys ni servicios de pago.**

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python app.py
```

Se abrirá la interfaz en el navegador. Con `share=True` se genera un link público temporal.

## Estructura

```
spvar_colombia_app/
├── app.py              # Aplicación principal (Gradio + SpVAR)
├── requirements.txt    # Dependencias
└── README.md           # Este archivo
```

## Licencia

Código de demostración académica. Datos sintéticos realistas.
