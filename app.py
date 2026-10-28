"""
SpVAR Colombia – Educación y Crecimiento Económico
Interfaz Gradio inmersiva + análisis espacial
Fuentes 100% públicas (sin API keys)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
import tempfile
import os

from libpysal.weights import Queen, Rook, KNN, lag_spatial
from esda.moran import Moran, Moran_Local
from esda.geary import Geary
from statsmodels.regression.linear_model import OLS as sm_OLS
from statsmodels.tools.tools import add_constant
from scipy.linalg import inv, eigvals
from scipy.interpolate import griddata

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")

# =============================================================================
# 1. CARGA DE GEOMETRÍA PÚBLICA
# =============================================================================
print("1/6 Cargando geometría pública de Colombia...")

URL_GEOJSON = (
    "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/"
    "co_2018_MGN_DPTO_POLITICO.geojson"
)
gdf = gpd.read_file(URL_GEOJSON)
gdf = gdf.rename(columns={"DPTO_CCDGO": "cod_dpto", "DPTO_CNMBR": "departamento"})
gdf["cod_dpto"] = gdf["cod_dpto"].astype(str).str.zfill(2)
gdf = gdf.to_crs("EPSG:4326")
print(f"   Departamentos cargados: {len(gdf)}")

# =============================================================================
# 2. PANEL SINTÉTICO LOCAL (sin APIs externas)
# =============================================================================
print("2/6 Generando panel sintético 2015-2024...")

np.random.seed(42)
N = len(gdf)
T = 10
years = list(range(2015, 2015 + T))

w_true = Queen.from_dataframe(gdf, use_index=False)
w_true.transform = "r"

educ = np.zeros((T, N))
growth = np.zeros((T, N))
viirs = np.zeros((T, N))

educ[0] = 8.2 + 1.8 * np.random.randn(N)
growth[0] = 2.8 + 1.4 * np.random.randn(N)
viirs[0] = 12 + 4.5 * np.random.randn(N)

for t in range(1, T):
    educ[t] = (
        0.62 * educ[t - 1]
        + 0.19 * lag_spatial(w_true, educ[t - 1])
        + 0.09 * growth[t - 1]
        + 0.35 * np.random.randn(N)
    )
    growth[t] = (
        0.51 * growth[t - 1]
        + 0.21 * lag_spatial(w_true, growth[t - 1])
        + 0.28 * educ[t - 1]
        + 0.14 * lag_spatial(w_true, educ[t - 1])
        + 0.85 * np.random.randn(N)
    )
    viirs[t] = (
        0.70 * viirs[t - 1]
        + 0.15 * lag_spatial(w_true, viirs[t - 1])
        + 0.40 * growth[t]
        + 1.1 * np.random.randn(N)
    )

df_long = pd.DataFrame(
    {
        "year": np.repeat(years, N),
        "departamento": np.tile(gdf["departamento"].values, T),
        "cod_dpto": np.tile(gdf["cod_dpto"].values, T),
        "educacion": educ.flatten(),
        "crecimiento": growth.flatten(),
        "viirs": viirs.flatten(),
    }
)

gdf["lon"] = gdf.geometry.centroid.x
gdf["lat"] = gdf.geometry.centroid.y
df_long = df_long.merge(gdf[["departamento", "lon", "lat"]], on="departamento", how="left")

# size debe ser > 0 en Plotly
min_growth = df_long["crecimiento"].min()
df_long["crecimiento_pos"] = df_long["crecimiento"] - min_growth + 1.5

print(f"   Observaciones: {len(df_long)} (N={N} × T={T})")

# =============================================================================
# 3. ESDA – Selección automática de W
# =============================================================================
print("3/6 Ejecutando ESDA y selección de matriz W...")


def build_weights(gdf):
    weights = {
        "Queen": Queen.from_dataframe(gdf, use_index=False),
        "Rook": Rook.from_dataframe(gdf, use_index=False),
    }
    for k in [3, 4, 5]:
        weights[f"KNN_{k}"] = KNN.from_dataframe(gdf, k=k, use_index=False)
    for name, w in weights.items():
        w.transform = "r"
    return weights


weights_dict = build_weights(gdf)
avg_educ = (
    df_long.groupby("departamento")["educacion"]
    .mean()
    .reindex(gdf["departamento"])
    .values
)
gdf["avg_educ"] = avg_educ


def select_best_W(gdf, y_var, weights_dict):
    results = []
    y = gdf[y_var].values
    for name, w in weights_dict.items():
        mi = Moran(y, w)
        results.append({"W": name, "Moran_I": mi.I, "p_value": mi.p_sim})
    res_df = pd.DataFrame(results)
    res_df = res_df[res_df["p_value"] < 0.05].copy()
    if res_df.empty:
        return weights_dict["Queen"], "Queen"
    res_df["abs_I"] = res_df["Moran_I"].abs()
    res_df = res_df.sort_values("abs_I", ascending=False)
    best_name = res_df.iloc[0]["W"]
    return weights_dict[best_name], best_name


best_W, best_W_name = select_best_W(gdf, "avg_educ", weights_dict)
print(f"   Matriz W seleccionada: {best_W_name}")

avg_data = (
    df_long.groupby("departamento")
    .agg({"educacion": "mean", "crecimiento": "mean", "viirs": "mean"})
    .reset_index()
)
gdf_map = gdf.merge(avg_data, on="departamento", how="left")

# =============================================================================
# 4. SpVAR – Estimación de los 4 mecanismos
# =============================================================================
print("4/6 Estimando SpVAR y los 4 mecanismos...")


def prepare_spvar_data(df_long, gdf, w):
    df = df_long.sort_values(["departamento", "year"]).copy()
    df["educ_l1"] = df.groupby("departamento")["educacion"].shift(1)
    df["growth_l1"] = df.groupby("departamento")["crecimiento"].shift(1)
    df = df.dropna(subset=["educ_l1", "growth_l1"]).reset_index(drop=True)
    years_av = sorted(df["year"].unique())
    rows = []
    for yr in years_av:
        sub = df[df["year"] == yr].set_index("departamento").reindex(gdf["departamento"])
        educ_l1 = sub["educ_l1"].values.astype(float)
        growth_l1 = sub["growth_l1"].values.astype(float)
        viirs_val = sub["viirs"].values.astype(float)
        W_educ = lag_spatial(w, educ_l1)
        W_growth = lag_spatial(w, growth_l1)
        for i, dep in enumerate(gdf["departamento"]):
            rows.append(
                {
                    "year": yr,
                    "departamento": dep,
                    "educacion": float(sub.loc[dep, "educacion"]),
                    "crecimiento": float(sub.loc[dep, "crecimiento"]),
                    "educ_l1": educ_l1[i],
                    "growth_l1": growth_l1[i],
                    "W_educ_l1": W_educ[i],
                    "W_growth_l1": W_growth[i],
                    "viirs": viirs_val[i],
                }
            )
    return pd.DataFrame(rows)


spvar_df = prepare_spvar_data(df_long, gdf, best_W)


def estimate_equation(y, X, var_names):
    X_c = add_constant(X)
    model = sm_OLS(y, X_c).fit(cov_type="HC1")
    coefs = dict(zip(["const"] + var_names, model.params))
    return coefs


coef_educ = estimate_equation(
    spvar_df["educacion"].values,
    spvar_df[["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"]].values,
    ["educ_l1", "growth_l1", "W_educ_l1", "W_growth_l1", "viirs"],
)
coef_growth = estimate_equation(
    spvar_df["crecimiento"].values,
    spvar_df[["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"]].values,
    ["growth_l1", "educ_l1", "W_growth_l1", "W_educ_l1", "viirs"],
)


def spatial_impacts(rho, beta, w):
    n = w.n
    S = inv(np.eye(n) - rho * w.full()[0])
    direct = np.mean(np.diag(S)) * beta
    total = np.mean(S.sum(axis=1)) * beta
    indirect = total - direct
    return {"direct": direct, "indirect": indirect, "total": total}


rho_e = coef_educ.get("W_educ_l1", 0.0)
rho_g = coef_growth.get("W_growth_l1", 0.0)
imp_e = spatial_impacts(rho_e, coef_educ["educ_l1"], best_W)
imp_g = spatial_impacts(rho_g, coef_growth["growth_l1"], best_W)

print("   Mecanismos e impactos calculados.")

# =============================================================================
# 5. Funciones de visualización para Gradio
# =============================================================================
print("5/6 Preparando funciones de visualización...")


def crear_mapa_coropletico():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#0f172a")
    gdf_map.plot(
        column="educacion",
        cmap="YlGnBu",
        linewidth=0.4,
        edgecolor="white",
        legend=True,
        ax=axes[0],
        legend_kwds={"shrink": 0.5},
    )
    axes[0].set_title("Educación promedio", fontweight="bold", color="white")
    axes[0].axis("off")
    gdf_map.plot(
        column="crecimiento",
        cmap="RdYlGn",
        linewidth=0.4,
        edgecolor="white",
        legend=True,
        ax=axes[1],
        legend_kwds={"shrink": 0.5},
    )
    axes[1].set_title("Crecimiento económico", fontweight="bold", color="white")
    axes[1].axis("off")
    gdf_map.plot(
        column="viirs",
        cmap="inferno",
        linewidth=0.4,
        edgecolor="white",
        legend=True,
        ax=axes[2],
        legend_kwds={"shrink": 0.5},
    )
    axes[2].set_title("Luces nocturnas (VIIRS)", fontweight="bold", color="white")
    axes[2].axis("off")
    plt.tight_layout()
    return fig


def crear_scatter_3d():
    fig = px.scatter_3d(
        df_long[df_long["year"] == 2024],
        x="lon",
        y="lat",
        z="viirs",
        color="educacion",
        size="crecimiento_pos",
        hover_name="departamento",
        color_continuous_scale="Turbo",
        size_max=22,
        title="Educación × Crecimiento × Luces (2024)",
    )
    fig.update_layout(
        template="plotly_dark",
        scene=dict(aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)),
        paper_bgcolor="#0f172a",
        font=dict(color="white"),
        height=520,
    )
    return fig


def crear_superficie_3d():
    gdf_2024 = df_long[df_long["year"] == 2024]
    lon = gdf_2024["lon"].values
    lat = gdf_2024["lat"].values
    z = gdf_2024["viirs"].values
    xi = np.linspace(lon.min(), lon.max(), 70)
    yi = np.linspace(lat.min(), lat.max(), 70)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((lon, lat), z, (Xi, Yi), method="cubic")
    fig = go.Figure(data=[go.Surface(x=Xi, y=Yi, z=Zi, colorscale="Inferno")])
    fig.update_layout(
        title="Superficie de intensidad de luces nocturnas",
        template="plotly_dark",
        scene=dict(
            xaxis_title="Longitud",
            yaxis_title="Latitud",
            zaxis_title="VIIRS",
            aspectmode="manual",
            aspectratio=dict(x=1.3, y=1, z=0.45),
        ),
        paper_bgcolor="#0f172a",
        font=dict(color="white"),
        height=520,
    )
    return fig


def crear_animacion_3d():
    fig = px.scatter_3d(
        df_long,
        x="lon",
        y="lat",
        z="viirs",
        color="educacion",
        size="crecimiento_pos",
        animation_frame="year",
        hover_name="departamento",
        color_continuous_scale="Turbo",
        size_max=18,
        title="Evolución temporal 2015-2024",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        scene=dict(aspectmode="manual", aspectratio=dict(x=1.2, y=1, z=0.55)),
        height=520,
    )
    return fig


def crear_mapa_folium_html():
    """Mapa Folium con tiles públicos CartoDB dark_matter"""
    m = folium.Map(location=[4.5, -74.0], zoom_start=5.3, tiles="CartoDB dark_matter")

    def add_layer(gdf, col, name, cmap):
        folium.Choropleth(
            geo_data=gdf.__geo_interface__,
            data=gdf,
            columns=["departamento", col],
            key_on="feature.properties.departamento",
            fill_color=cmap,
            fill_opacity=0.75,
            line_opacity=0.25,
            legend_name=name,
            name=name,
        ).add_to(m)

    add_layer(gdf_map, "viirs", "Luces nocturnas", "YlOrRd")
    add_layer(gdf_map, "educacion", "Educación", "YlGnBu")
    add_layer(gdf_map, "crecimiento", "Crecimiento", "RdYlGn")
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m)
    return m._repr_html_()


def resumen_mecanismos():
    return f"""
### Los 4 Mecanismos de Transmisión Espacial

**1. Autodinámica espacial** (inercia propia)  
• Educación → φ_ee = **{coef_educ['educ_l1']:.4f}**  
• Crecimiento → φ_gg = **{coef_growth['growth_l1']:.4f}**

**2. Cruce local** (misma región)  
• Educación ← Crecimiento local: **{coef_educ['growth_l1']:.4f}**  
• Crecimiento ← Educación local: **{coef_growth['educ_l1']:.4f}**

**3. Relación espacial directa** (vecinos → misma variable)  
• W·Educación → Educación: **{coef_educ['W_educ_l1']:.4f}**  
• W·Crecimiento → Crecimiento: **{coef_growth['W_growth_l1']:.4f}**

**4. Relación espacial cruzada** (vecinos → otra variable)  
• W·Crecimiento → Educación: **{coef_educ['W_growth_l1']:.4f}**  
• W·Educación → Crecimiento: **{coef_growth['W_educ_l1']:.4f}**

---
### Impactos de largo plazo (LeSage & Pace)

| Variable     | Directo   | Indirecto | Total    |
|--------------|-----------|-----------|----------|
| Educación    | {imp_e['direct']:.4f} | {imp_e['indirect']:.4f} | {imp_e['total']:.4f} |
| Crecimiento  | {imp_g['direct']:.4f} | {imp_g['indirect']:.4f} | {imp_g['total']:.4f} |

**Matriz W seleccionada:** `{best_W_name}`  
**Periodo:** {spvar_df['year'].min()} – {spvar_df['year'].max()}  
**Observaciones:** {len(spvar_df)}
"""


def generar_informe():
    contenido = f"""INFORME EJECUTIVO – SpVAR Educación y Crecimiento Económico
Colombia · Departamentos (2015-2024)

Fuente geometría: pública (GitHub raw – caticoa3/colombia_mapa)
Matriz de pesos espaciales: {best_W_name}
Observaciones: {len(spvar_df)}
Periodo: {spvar_df['year'].min()} – {spvar_df['year'].max()}

MECANISMOS DE TRANSMISIÓN
1. Autodinámica espacial
   Educación φ_ee = {coef_educ['educ_l1']:.4f}
   Crecimiento φ_gg = {coef_growth['growth_l1']:.4f}

2. Cruce local
   Educación ← Crecimiento = {coef_educ['growth_l1']:.4f}
   Crecimiento ← Educación = {coef_growth['educ_l1']:.4f}

3. Relación espacial directa
   W·Educ → Educ = {coef_educ['W_educ_l1']:.4f}
   W·Crec → Crec = {coef_growth['W_growth_l1']:.4f}

4. Relación espacial cruzada
   W·Crec → Educ = {coef_educ['W_growth_l1']:.4f}
   W·Educ → Crec = {coef_growth['W_educ_l1']:.4f}

IMPACTOS DE LARGO PLAZO (LeSage & Pace)
Educación  → Directo: {imp_e['direct']:.4f} | Indirecto: {imp_e['indirect']:.4f} | Total: {imp_e['total']:.4f}
Crecimiento → Directo: {imp_g['direct']:.4f} | Indirecto: {imp_g['indirect']:.4f} | Total: {imp_g['total']:.4f}
"""
    path = os.path.join(tempfile.gettempdir(), "informe_spvar_colombia.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(contenido)
    return path


# =============================================================================
# 6. INTERFAZ GRADIO
# =============================================================================
print("6/6 Construyendo interfaz Gradio...")

custom_css = """
.gradio-container { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate"),
    css=custom_css,
    title="SpVAR Colombia | Educación & Crecimiento",
) as demo:

    gr.Markdown(
        """
    # 🇨🇴 SpVAR Colombia
    ### Educación y Crecimiento Económico a nivel Departamental
    **Análisis espacial avanzado · 33 regiones · 2015-2024**  
    Fuentes 100% públicas · Sin API keys

    > Descubre cómo la educación de un departamento impacta el crecimiento de sus vecinos  
    > y cómo los spillovers espaciales configuran el desarrollo regional.
    """
    )

    with gr.Tabs():

        with gr.Tab("🗺️ Exploración Espacial"):
            gr.Markdown("### Distribución geográfica de las variables clave")
            btn_mapas = gr.Button("Generar mapas coropléticos", variant="primary")
            plot_mapas = gr.Plot(label="Mapas de alto espectro")
            btn_mapas.click(fn=crear_mapa_coropletico, outputs=plot_mapas)

            gr.Markdown("### Mapa interactivo multi-capa (tiles públicos CartoDB)")
            btn_folium = gr.Button("Cargar mapa Folium")
            html_folium = gr.HTML()
            btn_folium.click(fn=crear_mapa_folium_html, outputs=html_folium)

        with gr.Tab("🌌 Visualización 3D Inmersiva"):
            gr.Markdown("### Explora las dimensiones espaciales en 3D")
            with gr.Row():
                btn_3d = gr.Button("Scatter 3D 2024", variant="primary")
                btn_surf = gr.Button("Superficie de luces")
                btn_anim = gr.Button("Animación 2015-2024")
            plot_3d = gr.Plot(label="Visualización 3D")
            btn_3d.click(fn=crear_scatter_3d, outputs=plot_3d)
            btn_surf.click(fn=crear_superficie_3d, outputs=plot_3d)
            btn_anim.click(fn=crear_animacion_3d, outputs=plot_3d)

        with gr.Tab("📊 SpVAR & Mecanismos"):
            gr.Markdown(
                """
            ### Los cuatro canales de transmisión espacial
            El modelo SpVAR aísla cómo la educación y el crecimiento se transmiten  
            a través del tiempo y del espacio entre los departamentos de Colombia.
            """
            )
            btn_mec = gr.Button("Calcular y mostrar mecanismos", variant="primary")
            md_mecanismos = gr.Markdown()
            btn_mec.click(fn=resumen_mecanismos, outputs=md_mecanismos)

        with gr.Tab("📄 Informe Ejecutivo"):
            gr.Markdown(
                """
            ### Descarga tu informe personalizado
            Genera un resumen ejecutivo listo para compartir con stakeholders.
            """
            )
            btn_informe = gr.Button("Generar informe descargable", variant="primary")
            file_out = gr.File(label="Descargar informe (.txt)")
            btn_informe.click(fn=generar_informe, outputs=file_out)

            gr.Markdown(
                """
            ---
            **Fuente de geometría:** pública (GitHub raw)  
            **Datos:** sintéticos realistas con dependencia espacial  
            **Sin API keys ni servicios de pago**
            """
            )

    gr.Markdown(
        """
    <div style="text-align:center; margin-top:2rem; color:#94a3b8; font-size:0.85rem;">
    SpVAR Colombia · Modelo Vector Autorregresivo Espacial · Fuentes 100% públicas<br>
    Diseñado para experiencia inmersiva y storytelling de datos
    </div>
    """
    )

if __name__ == "__main__":
    print("Lanzando interfaz Gradio (share=True genera link público temporal)...")
    demo.launch(share=True, debug=False)
