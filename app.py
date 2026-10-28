"""
SpVAR Colombia – Educación y Crecimiento Económico
Streamlit Cloud friendly (sin geopandas/fiona/GDAL)
Fuentes 100% públicas · Sin API keys
"""

import warnings
warnings.filterwarnings("ignore")

import json
import urllib.request

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium

from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
from statsmodels.regression.linear_model import OLS as sm_OLS
from statsmodels.tools.tools import add_constant

st.set_page_config(
    page_title="SpVAR Colombia",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
h1, h2, h3, p, label, .stMarkdown { color: #e2e8f0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


def lag_spatial_knn(values, coords, k=4):
    """Retardo espacial aproximado con k vecinos más cercanos (row-standardized)."""
    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]
    out = np.zeros(len(values))
    for i in range(len(values)):
        out[i] = values[idx[i]].mean()
    return out


@st.cache_data(show_spinner="Cargando datos espaciales...")
def load_data():
    url = (
        "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/"
        "co_2018_MGN_DPTO_POLITICO.geojson"
    )
    with urllib.request.urlopen(url, timeout=60) as r:
        geojson = json.loads(r.read().decode())

    # Extraer nombres y centroides aproximados
    deps, lons, lats = [], [], []
    for feat in geojson["features"]:
        props = feat["properties"]
        name = props.get("DPTO_CNMBR") or props.get("departamento") or "N/A"
        deps.append(name)
        geom = feat["geometry"]
        # centroide simple del bounding box / promedio de coordenadas
        coords_list = []
        if geom["type"] == "Polygon":
            coords_list = geom["coordinates"][0]
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                coords_list.extend(poly[0])
        xs = [c[0] for c in coords_list]
        ys = [c[1] for c in coords_list]
        lons.append(float(np.mean(xs)))
        lats.append(float(np.mean(ys)))

    N = len(deps)
    coords = np.column_stack([lons, lats])

    # Panel sintético 2015-2024 con dependencia espacial
    np.random.seed(42)
    T = 10
    years = list(range(2015, 2015 + T))
    educ = np.zeros((T, N))
    growth = np.zeros((T, N))
    viirs = np.zeros((T, N))
    educ[0] = 8.2 + 1.8 * np.random.randn(N)
    growth[0] = 2.8 + 1.4 * np.random.randn(N)
    viirs[0] = 12 + 4.5 * np.random.randn(N)

    for t in range(1, T):
        educ[t] = (
            0.62 * educ[t - 1]
            + 0.19 * lag_spatial_knn(educ[t - 1], coords, k=4)
            + 0.09 * growth[t - 1]
            + 0.35 * np.random.randn(N)
        )
        growth[t] = (
            0.51 * growth[t - 1]
            + 0.21 * lag_spatial_knn(growth[t - 1], coords, k=4)
            + 0.28 * educ[t - 1]
            + 0.14 * lag_spatial_knn(educ[t - 1], coords, k=4)
            + 0.85 * np.random.randn(N)
        )
        viirs[t] = (
            0.70 * viirs[t - 1]
            + 0.15 * lag_spatial_knn(viirs[t - 1], coords, k=4)
            + 0.40 * growth[t]
            + 1.1 * np.random.randn(N)
        )

    df_long = pd.DataFrame(
        {
            "year": np.repeat(years, N),
            "departamento": np.tile(deps, T),
            "educacion": educ.flatten(),
            "crecimiento": growth.flatten(),
            "viirs": viirs.flatten(),
            "lon": np.tile(lons, T),
            "lat": np.tile(lats, T),
        }
    )
    df_long["crecimiento_pos"] = df_long["crecimiento"] - df_long["crecimiento"].min() + 1.5

    # Promedios por departamento (para mapas)
    avg = (
        df_long.groupby("departamento")
        .agg({"educacion": "mean", "crecimiento": "mean", "viirs": "mean", "lon": "first", "lat": "first"})
        .reset_index()
    )

    # Inyectar promedios en geojson para Folium
    avg_map = avg.set_index("departamento")
    for feat in geojson["features"]:
        name = feat["properties"].get("DPTO_CNMBR") or feat["properties"].get("departamento")
        if name in avg_map.index:
            feat["properties"]["educacion"] = float(avg_map.loc[name, "educacion"])
            feat["properties"]["crecimiento"] = float(avg_map.loc[name, "crecimiento"])
            feat["properties"]["viirs"] = float(avg_map.loc[name, "viirs"])
            feat["properties"]["departamento"] = name

    # SpVAR (retardos temporales + espaciales KNN)
    df = df_long.sort_values(["departamento", "year"]).copy()
    df["educ_l1"] = df.groupby("departamento")["educacion"].shift(1)
    df["growth_l1"] = df.groupby("departamento")["crecimiento"].shift(1)
    df = df.dropna(subset=["educ_l1", "growth_l1"]).reset_index(drop=True)

    rows = []
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr].set_index("departamento").reindex(deps)
        e1 = sub["educ_l1"].values.astype(float)
        g1 = sub["growth_l1"].values.astype(float)
        vv = sub["viirs"].values.astype(float)
        We = lag_spatial_knn(e1, coords, k=4)
        Wg = lag_spatial_knn(g1, coords, k=4)
        for i, dep in enumerate(deps):
            rows.append(
                {
                    "year": yr,
                    "departamento": dep,
                    "educacion": float(sub.loc[dep, "educacion"]),
                    "crecimiento": float(sub.loc[dep, "crecimiento"]),
                    "educ_l1": e1[i],
                    "growth_l1": g1[i],
                    "W_educ_l1": We[i],
                    "W_growth_l1": Wg[i],
                    "viirs": vv[i],
                }
            )
    spvar_df = pd.DataFrame(rows)

    def estimate(y, X, names):
        m = sm_OLS(y, add_constant(X)).fit(cov_type="HC1")
        return dict(zip(["const"] + names, m.params))

    coef_e = estimate(
        spvar_df["educacion"].values,
        spvar_df[["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"]].values,
        ["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"],
    )
    coef_g = estimate(
        spvar_df["crecimiento"].values,
        spvar_df[["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"]].values,
        ["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"],
    )

    # Impactos aproximados (multiplicador 1/(1-rho) estilo SAR simple)
    def impacts(rho, beta):
        rho = float(np.clip(rho, -0.9, 0.9))
        mult = 1.0 / (1.0 - rho) if abs(1.0 - rho) > 1e-6 else 1.0
        total = mult * beta
        direct = beta
        return {"direct": direct, "indirect": total - direct, "total": total}

    imp_e = impacts(coef_e.get("W_educ_l1", 0.0), coef_e["educ_l1"])
    imp_g = impacts(coef_g.get("W_growth_l1", 0.0), coef_g["growth_l1"])

    return geojson, avg, df_long, spvar_df, coef_e, coef_g, imp_e, imp_g, coords


geojson, avg, df_long, spvar_df, coef_educ, coef_growth, imp_e, imp_g, coords = load_data()

# ---- UI ----
st.title("🇨🇴 SpVAR Colombia")
st.subheader("Educación y Crecimiento Económico · 33 departamentos · 2015-2024")
st.caption("Fuentes 100% públicas · Sin API keys · Compatible Streamlit Cloud")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Exploración Espacial", "🌌 Visualización 3D", "📊 SpVAR & Mecanismos", "📄 Informe"]
)

with tab1:
    st.markdown("### Mapas coropléticos (promedios 2015-2024)")
    if st.button("Generar mapas", type="primary", key="m1"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        fig.patch.set_facecolor("#0f172a")
        for ax, col, title, cmap in zip(
            axes,
            ["educacion", "crecimiento", "viirs"],
            ["Educación", "Crecimiento", "Luces nocturnas"],
            ["YlGnBu", "RdYlGn", "inferno"],
        ):
            ax.set_facecolor("#0f172a")
            sc = ax.scatter(
                avg["lon"], avg["lat"], c=avg[col], cmap=cmap, s=120, edgecolors="white", linewidths=0.5
            )
            ax.set_title(title, color="white", fontweight="bold")
            ax.axis("off")
            fig.colorbar(sc, ax=ax, shrink=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("### Mapa interactivo Folium")
    if st.button("Cargar mapa Folium", key="m2"):
        m = folium.Map(location=[4.5, -74.0], zoom_start=5.3, tiles="CartoDB dark_matter")

        def add_choro(col, name, cmap):
            folium.Choropleth(
                geo_data=geojson,
                data=avg,
                columns=["departamento", col],
                key_on="feature.properties.departamento",
                fill_color=cmap,
                fill_opacity=0.75,
                line_opacity=0.25,
                legend_name=name,
                name=name,
            ).add_to(m)

        add_choro("viirs", "Luces nocturnas", "YlOrRd")
        add_choro("educacion", "Educación", "YlGnBu")
        add_choro("crecimiento", "Crecimiento", "RdYlGn")
        folium.LayerControl(collapsed=False).add_to(m)
        plugins.Fullscreen().add_to(m)
        st_folium(m, width=900, height=500)

with tab2:
    st.markdown("### Visualizaciones 3D")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Scatter 3D 2024", type="primary", key="b3d"):
            fig = px.scatter_3d(
                df_long[df_long["year"] == 2024],
                x="lon", y="lat", z="viirs",
                color="educacion", size="crecimiento_pos",
                hover_name="departamento",
                color_continuous_scale="Turbo", size_max=22,
                title="Educación × Crecimiento × Luces (2024)",
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0f172a", height=520,
                scene=dict(aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)),
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if st.button("Superficie luces", key="bsurf"):
            d = df_long[df_long["year"] == 2024]
            lon, lat, z = d["lon"].values, d["lat"].values, d["viirs"].values
            xi = np.linspace(lon.min(), lon.max(), 60)
            yi = np.linspace(lat.min(), lat.max(), 60)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = griddata((lon, lat), z, (Xi, Yi), method="cubic")
            fig = go.Figure(data=[go.Surface(x=Xi, y=Yi, z=Zi, colorscale="Inferno")])
            fig.update_layout(
                title="Superficie luces nocturnas", template="plotly_dark",
                paper_bgcolor="#0f172a", height=520,
                scene=dict(
                    xaxis_title="Lon", yaxis_title="Lat", zaxis_title="VIIRS",
                    aspectmode="manual", aspectratio=dict(x=1.3, y=1, z=0.45),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)
    with c3:
        if st.button("Animación 2015-2024", key="banim"):
            fig = px.scatter_3d(
                df_long, x="lon", y="lat", z="viirs",
                color="educacion", size="crecimiento_pos",
                animation_frame="year", hover_name="departamento",
                color_continuous_scale="Turbo", size_max=18,
                title="Evolución 2015-2024",
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0f172a", height=520,
                scene=dict(aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)),
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Cuatro mecanismos de transmisión espacial")
    if st.button("Mostrar mecanismos", type="primary", key="bmec"):
        st.markdown(
            f"""
**1. Autodinámica espacial**  
• Educación φ_ee = **{coef_educ['educ_l1']:.4f}**  
• Crecimiento φ_gg = **{coef_growth['growth_l1']:.4f}**

**2. Cruce local**  
• Educ ← Crec local: **{coef_educ['growth_l1']:.4f}**  
• Crec ← Educ local: **{coef_growth['educ_l1']:.4f}**

**3. Relación espacial directa** (KNN vecinos)  
• W·Educ → Educ: **{coef_educ['W_educ_l1']:.4f}**  
• W·Crec → Crec: **{coef_growth['W_growth_l1']:.4f}**

**4. Relación espacial cruzada**  
• W·Crec → Educ: **{coef_educ['W_growth_l1']:.4f}**  
• W·Educ → Crec: **{coef_growth['W_educ_l1']:.4f}**

---
### Impactos de largo plazo (aproximación)

| Variable | Directo | Indirecto | Total |
|----------|---------|-----------|-------|
| Educación | {imp_e['direct']:.4f} | {imp_e['indirect']:.4f} | {imp_e['total']:.4f} |
| Crecimiento | {imp_g['direct']:.4f} | {imp_g['indirect']:.4f} | {imp_g['total']:.4f} |

**W:** KNN k=4 · **Obs:** {len(spvar_df)} · **Periodo:** {spvar_df['year'].min()}–{spvar_df['year'].max()}
"""
        )

with tab4:
    st.markdown("### Informe ejecutivo")
    informe = f"""INFORME EJECUTIVO – SpVAR Educación y Crecimiento
Colombia · Departamentos 2015-2024

Fuente geometría: GitHub raw (caticoa3/colombia_mapa)
Matriz W: KNN k=4
Observaciones: {len(spvar_df)}
Periodo: {spvar_df['year'].min()} – {spvar_df['year'].max()}

MECANISMOS
1. Autodinámica
   Educ φ_ee = {coef_educ['educ_l1']:.4f}
   Crec φ_gg = {coef_growth['growth_l1']:.4f}
2. Cruce local
   Educ ← Crec = {coef_educ['growth_l1']:.4f}
   Crec ← Educ = {coef_growth['educ_l1']:.4f}
3. Espacial directa
   W·Educ → Educ = {coef_educ['W_educ_l1']:.4f}
   W·Crec → Crec = {coef_growth['W_growth_l1']:.4f}
4. Espacial cruzada
   W·Crec → Educ = {coef_educ['W_growth_l1']:.4f}
   W·Educ → Crec = {coef_growth['W_educ_l1']:.4f}

IMPACTOS
Educación  Directo={imp_e['direct']:.4f} Indirecto={imp_e['indirect']:.4f} Total={imp_e['total']:.4f}
Crecimiento Directo={imp_g['direct']:.4f} Indirecto={imp_g['indirect']:.4f} Total={imp_g['total']:.4f}
"""
    st.download_button(
        "📥 Descargar informe (.txt)",
        data=informe,
        file_name="informe_spvar_colombia.txt",
        mime="text/plain",
        type="primary",
    )
    st.caption("Fuente geometría pública · Datos sintéticos · Sin API keys")

st.markdown(
    '<div style="text-align:center;margin-top:2rem;color:#94a3b8;font-size:0.85rem;">'
    "SpVAR Colombia · Compatible Streamlit Cloud · Fuentes 100% públicas</div>",
    unsafe_allow_html=True,
)
