# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4.0",
#     "astrophot>=0.16.0",
#     "astropy>=6.0.0",
#     "duckdb>=1.1.0",
#     "marimo",
#     "numpy>=1.26.4",
#     "openai>=1.50.0",
#     "plotly[express,kaleido]>=5.24.0",
#     "polars[pyarrow]>=1.12.0",
#     "torch>=2.5.0",
#     "torch-geometric>=2.6.0",
# ]
# ///

import marimo

__generated_with = "0.14.13"
app = marimo.App(width="columns", sql_output="native")

with app.setup:
    import altair as alt
    import duckdb
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.table import Table



@app.cell
def _():
    from astrophot.models import AstroPhot_Model as AstroPhotModel
    from astrophot.image import Target_Image as TargetImage
    return AstroPhotModel, TargetImage


@app.cell
def _(mo):
    mo.md("""
    # 🌌 Astrophotography Simulation
    
    Simulate astronomical objects using **AstroPhot**.
    """)
    return


@app.cell
def _(TargetImage):
    basic_target_image = TargetImage(data=np.zeros((100, 100)), pixelscale=1, zeropoint=20)
    return (basic_target_image,)


@app.cell
def _(AstroPhotModel, basic_target_image, mo):
    mo.md("## 🔭 Model Configuration")

    center_x = mo.ui.slider(start=0, stop=100, value=50, label="Center X")
    center_y = mo.ui.slider(start=0, stop=100, value=50, label="Center Y")
    flux = mo.ui.number(start=0.1, stop=10.0, value=1.0, step=0.1, label="Flux")

    parameters = mo.vstack([center_x, center_y, flux])
    
    # Create model dynamically
    flat_sky_model = AstroPhotModel(
        model_type="flat sky model", 
        parameters={"center": [center_x.value, center_y.value], "F": flux.value}, 
        target=basic_target_image
    )
    flat_sky_model.initialize()

    mo.hstack([
        parameters,
        mo.md(f"""
        **Model Parameters:**
        - Center: ({center_x.value}, {center_y.value})
        - Flux: {flux.value}
        
        **Parameter Order:** {flat_sky_model.parameter_order}
        """)
    ])
    return center_x, center_y, flat_sky_model, flux, parameters


if __name__ == "__main__":
    app.run()
