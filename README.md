# SpVAR Colombia

App Streamlit: Educación y Crecimiento Económico (modelo SpVAR).

## Streamlit Cloud

1. Sube a GitHub **solo estos 3 archivos** en la raíz:
   - `app.py`
   - `requirements.txt`
   - `README.md`
2. [share.streamlit.io](https://share.streamlit.io) → New app
3. Main file: `app.py`
4. Deploy

## Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fuentes públicas

- GeoJSON: https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson
- Tiles: CartoDB dark_matter
- Datos: sintéticos locales
