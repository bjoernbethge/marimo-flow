# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4.0",
#     "astropy[recommended]>=6.0.0",
#     "duckdb>=1.1.0",
#     "marimo",
#     "nbformat>=5.10.0",
#     "numpy>=1.26.4",
#     "plotly>=5.24.0",
#     "polars>=1.12.0",
#     "vegafusion>=1.6.0",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium", sql_output="native")


@app.cell
def _():
    import pickle
    from io import BytesIO
    from pathlib import Path

    import altair as alt
    import duckdb
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.table import Table
    from plotly.subplots import make_subplots
    return (
        BytesIO, Path, Table, alt, duckdb, fits, go, make_subplots,
        mo, np, pickle, pl, px, u
    )


@app.cell
def _(mo):
    mo.md("""
    # 🌌 COSMOS-Web DR1 Catalog Analysis

    Interactive analysis of the COSMOS-Web galaxy catalog with **784,016 celestial objects**
    from JWST, HST, and ground-based observations.

    ### Features of this notebook:
    - 📊 Interactive visualizations with Plotly Express & Altair
    - 🚀 High-performance data processing with Polars
    - 🦆 SQL queries with DuckDB
    - 🔍 Reactive data filtering and exploration
    """)
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".fits"],
        label="📁 Upload COSMOS-Web catalog (COSMOSWeb_mastercatalog_v1.fits)",
        multiple=False,
    )

    pdfz_upload = mo.ui.file(
        filetypes=[".pkl"],
        label="📁 Upload PDF(z) file (optional)",
        multiple=False
    )

    mo.vstack([
        mo.md("### Load Data"),
        file_upload,
        pdfz_upload
    ])
    return file_upload, pdfz_upload


@app.function
def fits_to_polars(hdu_data):
    """Convert FITS HDU directly to Polars DataFrame"""
    import numpy as np
    import polars as pl
    
    numpy_array = np.array(hdu_data)

    columns = {}
    for name in numpy_array.dtype.names:
        columns[name] = numpy_array[name]

    return pl.DataFrame(columns)


@app.cell
def _(BytesIO, file_upload, fits, mo, np, pdfz_upload, pickle, pl):
    # Global variables
    catalog_loaded = False
    cat_photom = None
    cat_lephare = None
    cat_cigale = None
    cat_bd = None
    lephare_pdfz = None

    if file_upload.value:
        with mo.status.progress_bar(total=5) as bar:
            bar.update(title="Opening catalog...")

            # Read FITS from bytes
            file_contents = file_upload.value[0].contents

            with fits.open(BytesIO(file_contents), memmap=True) as hdu:
                bar.update(title="Loading photometry...")
                cat_photom = fits_to_polars(hdu[1].data)

                bar.update(title="Loading LePhare...")
                cat_lephare = fits_to_polars(hdu[2].data)

                bar.update(title="Loading CIGALE...")
                cat_cigale = fits_to_polars(hdu[4].data)
                cat_bd = fits_to_polars(hdu[6].data)

            bar.update(title="Optimizing data types...")
            # Float64 -> Float32 for memory efficiency
            for df_name, df in [("photom", cat_photom), ("lephare", cat_lephare)]:
                float_cols = [col for col in df.columns if df[col].dtype == pl.Float64]
                if float_cols:
                    if df_name == "photom":
                        cat_photom = df.with_columns([
                            pl.col(c).cast(pl.Float32) for c in float_cols
                        ])
                    else:
                        cat_lephare = df.with_columns([
                            pl.col(c).cast(pl.Float32) for c in float_cols
                        ])

        catalog_loaded = True

        # Load PDF(z) if available
        if pdfz_upload.value:
            pdfz_contents = pdfz_upload.value[0].contents
            lephare_pdfz = pickle.loads(pdfz_contents)

        mo.md(f"""
        ✅ **Catalog successfully loaded!**
        - Photometry: {len(cat_photom):,} objects, {len(cat_photom.columns)} columns
        - LePhare: {len(cat_lephare.columns)} parameters
        - CIGALE: {len(cat_cigale.columns)} parameters
        - PDF(z): {'✅ Loaded' if lephare_pdfz else '❌ Not loaded'}
        """)
    else:
        mo.stop(True, mo.md("⏳ Please upload the COSMOS-Web catalog file..."))

    return (
        catalog_loaded, cat_bd, cat_cigale, cat_lephare,
        cat_photom, lephare_pdfz
    )


@app.cell
def _(catalog_loaded, mo):
    mo.stop(not catalog_loaded, mo.md(""))

    # Filter UI components
    quality_filter = mo.ui.checkbox(
        value=True,
        label="High-quality sources only (warn_flag = 0)"
    )

    galaxy_only = mo.ui.checkbox(
        value=True,
        label="Galaxies only (type = 0)"
    )

    mag_limit = mo.ui.slider(
        start=20,
        stop=30,
        value=25,
        step=0.5,
        label="F444W Magnitude limit"
    )

    z_range = mo.ui.range_slider(
        start=0,
        stop=12,
        value=[0, 6],
        step=0.1,
        label="Redshift range"
    )

    mo.vstack([
        mo.md("### 🔧 Data Filters"),
        mo.hstack([quality_filter, galaxy_only]),
        mag_limit,
        z_range
    ])
    return galaxy_only, mag_limit, quality_filter, z_range


@app.cell
def _(cat_lephare, cat_photom, catalog_loaded, galaxy_only, mag_limit, mo, pl, quality_filter, z_range):
    mo.stop(not catalog_loaded, mo.md(""))

    # Apply base filters with Polars
    filtered_data = cat_photom

    if quality_filter.value:
        filtered_data = filtered_data.filter(pl.col("warn_flag") == 0)

    if galaxy_only.value:
        # Join with LePhare for type information
        filtered_data = filtered_data.join(
            cat_lephare.select(["id", "type", "zfinal", "mass_med", "ssfr_med"]),
            on="id",
            how="inner"
        ).filter(pl.col("type") == 0)
    else:
        filtered_data = filtered_data.join(
            cat_lephare.select(["id", "zfinal", "mass_med", "ssfr_med"]),
            on="id",
            how="inner"
        )

    # Magnitude and redshift filters
    filtered_data = filtered_data.filter(
        (pl.col("mag_model_f444w").abs() < mag_limit.value) &
        (pl.col("zfinal") >= z_range.value[0]) &
        (pl.col("zfinal") <= z_range.value[1])
    )

    mo.md(f"""
    ### 📊 Filtered Data
    - **{len(filtered_data):,}** of {len(cat_photom):,} objects selected
    - **{len(filtered_data) / len(cat_photom) * 100:.1f}%** of total data
    """)
    return filtered_data,


@app.cell
def _(catalog_loaded, filtered_data, mo, px):
    mo.stop(not catalog_loaded or len(filtered_data) == 0, mo.md(""))

    # Sample for performance
    sample_size = min(10000, len(filtered_data))
    spatial_sample = filtered_data.sample(n=sample_size, seed=42)

    fig_spatial = px.density_hexbin(
        spatial_sample.to_pandas(),
        x="ra",
        y="dec",
        nbinsx=80,
        nbinsy=80,
        color_continuous_scale="Viridis",
        title=f"Spatial Distribution ({sample_size:,} objects)",
        labels={"ra": "Right Ascension [°]", "dec": "Declination [°]"}
    )

    fig_spatial.update_layout(
        height=500,
        template="plotly_dark",
        coloraxis_colorbar_title="Count"
    )

    mo.ui.plotly(fig_spatial)
    return fig_spatial, sample_size, spatial_sample


@app.cell
def _(alt, catalog_loaded, filtered_data, mo):
    mo.stop(not catalog_loaded or len(filtered_data) == 0, mo.md(""))

    # Prepare histogram data
    z_hist = filtered_data.select("zfinal").to_pandas()

    # Interactive Altair chart
    z_chart = alt.Chart(z_hist).mark_bar(
        opacity=0.8,
        color="steelblue"
    ).encode(
        alt.X("zfinal:Q",
              bin=alt.Bin(maxbins=50),
              title="Redshift z"),
        alt.Y("count()",
              title="Number of Galaxies"),
        tooltip=[
            alt.Tooltip("zfinal:Q", title="z", format=".2f"),
            alt.Tooltip("count()", title="Count")
        ]
    ).properties(
        width=700,
        height=300,
        title="Redshift Distribution"
    ).interactive()

    mo.ui.altair_chart(z_chart)
    return z_chart, z_hist


@app.cell
def _(alt, cat_lephare, catalog_loaded, filtered_data, mo, pl):
    mo.stop(not catalog_loaded or len(filtered_data) == 0, mo.md(""))

    # Calculate colors
    color_data = filtered_data.join(
        cat_lephare.select(["id", "mabs_nuv", "mabs_r", "mabs_j"]),
        on="id",
        how="inner"
    ).with_columns([
        (pl.col("mabs_nuv") - pl.col("mabs_r")).alias("nuv_r"),
        (pl.col("mabs_r") - pl.col("mabs_j")).alias("r_j")
    ]).filter(
        pl.col("nuv_r").is_not_null() &
        pl.col("r_j").is_not_null()
    )

    # Sample for performance
    color_sample = color_data.sample(n=min(5000, len(color_data)), seed=42)

    # Interactive scatter plot with Altair
    brush = alt.selection_interval()

    color_chart = alt.Chart(color_sample.to_pandas()).mark_circle(
        size=30,
        opacity=0.6
    ).encode(
        x=alt.X("r_j:Q",
                scale=alt.Scale(domain=[-1, 3]),
                title="r - J"),
        y=alt.Y("nuv_r:Q",
                scale=alt.Scale(domain=[-1, 9]),
                title="NUV - r"),
        color=alt.Color("zfinal:Q",
                       scale=alt.Scale(scheme="viridis"),
                       title="Redshift"),
        tooltip=[
            alt.Tooltip("id:N", title="ID"),
            alt.Tooltip("zfinal:Q", title="z", format=".2f"),
            alt.Tooltip("mass_med:Q", title="log M*", format=".2f"),
            alt.Tooltip("nuv_r:Q", title="NUV-r", format=".2f"),
            alt.Tooltip("r_j:Q", title="r-J", format=".2f")
        ]
    ).add_params(
        brush
    ).properties(
        width=600,
        height=500,
        title="NUV-r-J Color-Color Diagram"
    )

    # Display selected data
    selected_chart = mo.ui.altair_chart(color_chart)

    mo.vstack([
        selected_chart,
        mo.md("### 🎯 Selected Galaxies"),
        mo.ui.table(
            selected_chart.value.select(["id", "zfinal", "mass_med", "nuv_r", "r_j"]).head(10)
            if len(selected_chart.value) > 0 else None,
            pagination=False
        )
    ])
    return brush, color_chart, color_data, color_sample, selected_chart


@app.cell
def _(catalog_loaded, filtered_data, mo, pl, px):
    mo.stop(not catalog_loaded or len(filtered_data) == 0, mo.md(""))

    mass_z_data = filtered_data.filter(
        pl.col("mass_med").is_not_null() &
        (pl.col("mass_med") > 8)
    )

    # 2D histogram with Plotly
    fig_mass_z = px.density_heatmap(
        mass_z_data.to_pandas(),
        x="zfinal",
        y="mass_med",
        nbinsx=50,
        nbinsy=30,
        color_continuous_scale="Plasma",
        title="Stellar Mass vs. Redshift",
        labels={
            "zfinal": "Redshift z",
            "mass_med": "log(M*/M☉)",
            "count": "Count"
        }
    )

    fig_mass_z.update_layout(
        height=500,
        template="plotly_dark"
    )

    mo.ui.plotly(fig_mass_z)
    return fig_mass_z, mass_z_data


@app.cell
def _(cat_cigale, cat_lephare, cat_photom, catalog_loaded, duckdb, mo):
    mo.stop(not catalog_loaded, mo.md(""))

    mo.md("""
    ### 🦆 DuckDB Integration for SQL Queries

    Harness the power of DuckDB for complex astronomical queries!
    """)

    # Create DuckDB connection
    con = duckdb.connect(':memory:')

    # Register Polars DataFrames in DuckDB
    con.register('photom', cat_photom.to_arrow())
    con.register('lephare', cat_lephare.to_arrow())
    con.register('cigale', cat_cigale.to_arrow())

    # SQL Query Editor
    sql_query = mo.ui.text_area(
        value="""-- Example: Top 10 massive galaxies at z > 3
SELECT
    p.id,
    l.zfinal as z,
    l.mass_med as log_mass,
    p.mag_model_f444w as mag_f444w,
    p.ra,
    p.dec
FROM photom p
JOIN lephare l ON p.id = l.id
WHERE l.zfinal > 3.0
    AND l.mass_med > 10.5
    AND p.warn_flag = 0
ORDER BY l.mass_med DESC
LIMIT 10""",
        label="SQL Query",
        rows=10
    )

    run_query = mo.ui.button(label="🚀 Execute Query", kind="primary")

    mo.vstack([sql_query, run_query])
    return con, run_query, sql_query


@app.cell
def _(catalog_loaded, con, mo, run_query, sql_query):
    mo.stop(not catalog_loaded or not run_query.value, mo.md(""))

    try:
        # Execute query
        result = con.execute(sql_query.value).pl()

        mo.vstack([
            mo.md(f"### 📊 Query Results ({len(result)} rows)"),
            mo.ui.table(result.head(100), pagination=True),
            mo.ui.download(
                data=result.write_csv().encode(),
                filename="query_results.csv",
                label="📥 Export as CSV"
            )
        ])
    except Exception as e:
        mo.md(f"❌ **Error:** {e!s}")
    return result,

@app.cell
def _(catalog_loaded, mo):
    mo.stop(not catalog_loaded, mo.md(""))

    mo.md("""
    ### 📚 Additional Information

    #### Data Quality Flags:
    - `warn_flag = 0`: Highest quality
    - `warn_flag = 1`: Hot pixels
    - `warn_flag = 2-6`: Various quality issues

    #### Available Catalogs:
    1. **Photometry**: PSF-homogenized aperture and model photometry
    2. **LePhare**: Photometric redshifts and physical parameters
    3. **CIGALE**: Detailed SED fits and star formation histories
    4. **ML Morphology**: Machine learning-based morphological classification

    #### Performance Tips:
    - Use filters to reduce the data volume
    - Samples are automatically created for large visualizations
    - Polars provides very fast data processing
    - DuckDB is excellent for complex SQL queries

    ---
    *This notebook uses Marimo's reactive features for interactive data exploration.*
    """)
    return


if __name__ == "__main__":
    app.run()
