"""
Plotly Interactive Visualizations - 3D, Subplots, Animations

Based on refs/plotly-quickstart.md
Demonstrates:
- 3D scatter and surface plots
- Subplots and multiple traces
- Interactive features (hover, zoom, pan)
- Animations and sliders
- Custom themes and styling
"""

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Plotly Interactive Visualizations

        **Plotly** enables rich, interactive charts with hover, zoom, and animations.

        ## Chart Types
        - 📊 3D Scatter & Surface Plots
        - 📈 Subplots with Multiple Traces
        - 🎬 Animations with Sliders
        - 🎨 Custom Themes & Styling
        """
    )
    return


@app.cell
def _():
    """Import libraries"""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import polars as pl
    import numpy as np
    from sklearn.datasets import load_iris, make_classification

    return go, make_classification, make_subplots, np, pl, px


@app.cell
def _(mo):
    """Chart type selection"""
    chart_type = mo.ui.dropdown(
        options=["3D Scatter", "3D Surface", "Subplots", "Animation", "Custom Styling"],
        value="3D Scatter",
        label="📊 Chart Type",
    )
    mo.md(f"## Chart Selection\n\n{chart_type}")
    return (chart_type,)


@app.cell
def _(chart_type, go, load_iris, make_classification, make_subplots, np, pl, px):
    """Generate visualization based on selection"""

    if chart_type.value == "3D Scatter":
        # 3D scatter plot with iris dataset
        iris = load_iris()
        df = pl.DataFrame(iris.data, schema=iris.feature_names)
        df = df.with_columns(
            pl.Series("species", [iris.target_names[i] for i in iris.target])
        )

        fig = px.scatter_3d(
            df.to_pandas(),
            x="sepal length (cm)",
            y="sepal width (cm)",
            z="petal length (cm)",
            color="species",
            size="petal width (cm)",
            hover_data=["petal width (cm)"],
            title="3D Iris Dataset Visualization",
            labels={"species": "Species"},
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="Sepal Length",
                yaxis_title="Sepal Width",
                zaxis_title="Petal Length",
            ),
            height=600,
        )

    elif chart_type.value == "3D Surface":
        # 3D surface plot
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))

        fig = go.Figure(
            data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis", showscale=True)]
        )

        fig.update_layout(
            title="3D Surface: sin(√(x² + y²))",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            ),
            height=600,
        )

    elif chart_type.value == "Subplots":
        # Create multiple subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Scatter", "Bar Chart", "Line Chart", "Pie Chart"),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "pie"}],
            ],
        )

        # Subplot 1: Scatter
        fig.add_trace(
            go.Scatter(
                x=np.random.randn(100),
                y=np.random.randn(100),
                mode="markers",
                name="Scatter",
                marker=dict(color="blue", size=8),
            ),
            row=1,
            col=1,
        )

        # Subplot 2: Bar
        categories = ["A", "B", "C", "D"]
        values = [23, 45, 56, 78]
        fig.add_trace(
            go.Bar(x=categories, y=values, name="Bar", marker_color="green"), row=1, col=2
        )

        # Subplot 3: Line
        x = np.linspace(0, 10, 100)
        fig.add_trace(
            go.Scatter(x=x, y=np.sin(x), mode="lines", name="Sin", line=dict(color="red")),
            row=2,
            col=1,
        )

        # Subplot 4: Pie
        fig.add_trace(
            go.Pie(labels=categories, values=values, name="Pie"), row=2, col=2
        )

        fig.update_layout(height=700, title_text="Multiple Subplots Dashboard", showlegend=True)

    elif chart_type.value == "Animation":
        # Animated scatter plot
        np.random.seed(42)
        n_frames = 20
        n_points = 50

        frames = []
        for i in range(n_frames):
            x = np.random.randn(n_points) + i * 0.2
            y = np.random.randn(n_points) + np.sin(i / 3)
            frames.append({"x": x, "y": y, "frame": i})

        df_anim = pl.DataFrame(frames)

        fig = px.scatter(
            df_anim.to_pandas(),
            x="x",
            y="y",
            animation_frame="frame",
            range_x=[-3, 10],
            range_y=[-4, 4],
            title="Animated Scatter Plot",
        )

        fig.update_layout(height=600)
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200

    else:  # Custom Styling
        # Styled chart with custom theme
        X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_clusters_per_class=1)

        df = pl.DataFrame({"x": X[:, 0], "y": X[:, 1], "class": y.astype(str)})

        fig = px.scatter(
            df.to_pandas(),
            x="x",
            y="y",
            color="class",
            title="Custom Styled Chart",
            labels={"class": "Class"},
            color_discrete_sequence=["#FF6B6B", "#4ECDC4"],
        )

        # Custom styling
        fig.update_layout(
            template="plotly_dark",
            font=dict(family="Arial, sans-serif", size=14, color="#FFFFFF"),
            title=dict(font=dict(size=24, color="#4ECDC4")),
            paper_bgcolor="#1E1E1E",
            plot_bgcolor="#2D2D2D",
            height=600,
        )

        fig.update_traces(marker=dict(size=12, opacity=0.7, line=dict(width=1, color="white")))

    chart = fig
    return (chart,)


@app.cell
def _(chart, mo):
    """Display chart"""
    mo.md(
        f"""
        ## 📊 Interactive Visualization

        {mo.ui.plotly(chart)}

        **Interactive Features:**
        - 🖱️ Hover for details
        - 🔍 Click and drag to zoom
        - 📐 Double-click to reset
        - 🎯 Click legend to toggle traces
        """
    )
    return


@app.cell
def _(mo):
    """Code examples"""
    mo.md(
        r"""
        ## 💻 Code Patterns

        ### 3D Scatter
        ```python
        import plotly.express as px
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='category')
        ```

        ### Subplots
        ```python
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=2)
        fig.add_trace(go.Scatter(...), row=1, col=1)
        ```

        ### Animation
        ```python
        fig = px.scatter(df, x='x', y='y', animation_frame='time')
        ```

        ### Custom Theme
        ```python
        fig.update_layout(
            template='plotly_dark',
            font=dict(family='Arial', size=14)
        )
        ```

        ## 📚 Learn More
        - See `refs/plotly-quickstart.md` for comprehensive patterns
        - [Plotly Python Documentation](https://plotly.com/python/)
        """
    )
    return


if __name__ == "__main__":
    app.run()
