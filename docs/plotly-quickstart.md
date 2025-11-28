# Plotly Reference - Quickstart & Core Concepts

**Last Updated**: 2025-11-21
**Source Version**: Latest (plotly/plotly.py)
**Status**: Current

## What is Plotly?

Plotly is an open-source, browser-based interactive graphing library for Python. It provides:

- **Interactive Visualizations**: Hover info, zoom, pan, toggle series
- **Declarative API**: Simple, intuitive syntax for creating charts
- **30+ Chart Types**: Scatter, bar, line, 3D, statistical, financial, and more
- **Export Options**: Save as HTML, PNG, SVG, or publish online
- **Customizable**: Extensive layout and styling options

Two main approaches:
1. **Plotly Express** (px) - High-level, simple syntax for common charts
2. **Graph Objects** (go) - Low-level, full control for complex visualizations

## Quick Reference

### Installation

```bash
# Basic installation
pip install plotly

# With chart-studio for online features
pip install plotly chart-studio

# Using uv
uv add plotly

# Using conda
conda install -c plotly plotly
```

### Basic Chart in 3 Lines

```python
import plotly.express as px

fig = px.scatter(data_frame, x="column_x", y="column_y")
fig.show()
```

## Core Concepts

### 1. Plotly Express - High-Level API

Plotly Express is the recommended starting point for most use cases.

#### Basic Scatter Plot

```python
import plotly.express as px
import pandas as pd

# With DataFrame
df = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [2, 4, 3, 5],
    "category": ["A", "A", "B", "B"]
})

fig = px.scatter(df, x="x", y="y")
fig.show()
```

#### Scatter with Color Coding

```python
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="category",  # Color by category
    size="y",          # Size proportional to y
    hover_name="category"  # Show on hover
)
fig.show()
```

#### Line Chart

```python
fig = px.line(
    df,
    x="date",
    y="value",
    title="Time Series"
)
fig.show()
```

#### Bar Chart

```python
fig = px.bar(
    df,
    x="category",
    y="value",
    color="group",  # Grouped bars
    title="Sales by Category"
)
fig.show()
```

#### Histogram

```python
fig = px.histogram(
    df,
    x="value",
    nbins=20,
    title="Distribution"
)
fig.show()
```

#### Box Plot

```python
fig = px.box(
    df,
    x="category",
    y="value",
    title="Distribution by Category"
)
fig.show()
```

#### Heatmap

```python
fig = px.imshow(
    data_matrix,
    labels=dict(color="Value"),
    title="Heatmap"
)
fig.show()
```

### 2. Graph Objects - Low-Level API

For more control and complex visualizations, use Graph Objects directly.

#### Basic Figure with Traces

```python
import plotly.graph_objects as go

fig = go.Figure(
    data=[
        go.Scatter(x=[1, 2, 3], y=[2, 4, 3], name="Series 1"),
        go.Scatter(x=[1, 2, 3], y=[3, 5, 4], name="Series 2"),
    ],
    layout=go.Layout(
        title="My Chart",
        xaxis_title="X Axis",
        yaxis_title="Y Axis"
    )
)
fig.show()
```

#### Building with add_trace()

```python
import plotly.graph_objects as go

fig = go.Figure()

# Add scatter trace
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17],
    mode='lines+markers',
    name='Line 1'
))

# Add bar trace
fig.add_trace(go.Bar(
    x=[1, 2, 3, 4],
    y=[5, 6, 4, 7],
    name='Bars'
))

fig.update_layout(title="Combined Chart")
fig.show()
```

#### Horizontal Bar Chart

```python
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Bar(
    y=['Product A', 'Product B', 'Product C'],
    x=[20, 14, 23],
    orientation='h',
    name='Sales',
    marker=dict(color='lightblue', line=dict(color='navy', width=2))
))

fig.update_layout(
    title="Sales by Product",
    xaxis_title="Sales Amount",
    yaxis_title="Product"
)
fig.show()
```

### 3. Figure Properties & Updates

#### Update Layout (Axes, Titles, Legend)

```python
import plotly.graph_objects as go

fig = go.Figure(...)

# Update overall layout
fig.update_layout(
    title="Main Title",
    xaxis_title="X Axis Label",
    yaxis_title="Y Axis Label",
    hovermode='x unified',  # Hover mode
    template='plotly_dark',  # Theme
    width=1200,
    height=600,
    font=dict(size=12, family="Arial")
)

fig.show()
```

#### Update Axes

```python
# Log scale
fig.update_xaxes(type="log")

# Date format
fig.update_xaxes(type="date")

# Range
fig.update_yaxes(range=[0, 100])

# Grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
```

#### Update Traces

```python
# Update specific trace
fig.data[0].update(
    marker=dict(size=10, color='red'),
    line=dict(width=2)
)

# Update all traces
fig.update_traces(
    marker=dict(size=8),
    hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
)
```

#### Update Legend

```python
fig.update_layout(
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='black',
        borderwidth=1
    )
)
```

### 4. Customization & Styling

#### Colors

```python
import plotly.express as px

# Using color mapping
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="category",
    color_discrete_map={"A": "red", "B": "blue"}
)

# Custom color scale for continuous data
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="value",
    color_continuous_scale="Viridis"  # or "Blues", "Reds", etc.
)
fig.show()
```

#### Templates/Themes

```python
# Available templates
fig.update_layout(template="plotly")           # Default
fig.update_layout(template="plotly_dark")      # Dark theme
fig.update_layout(template="plotly_white")     # Clean white
fig.update_layout(template="ggplot2")          # ggplot2 style
fig.update_layout(template="seaborn")          # Seaborn style

fig.show()
```

#### Markers & Lines

```python
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 15, 13, 17],
    mode='lines+markers',
    line=dict(
        color='rgb(0, 0, 255)',
        width=3,
        dash='dash'  # 'solid', 'dot', 'dash', 'longdash'
    ),
    marker=dict(
        size=12,
        symbol='circle-open',  # 'circle', 'square', 'diamond', etc.
        color='red',
        line=dict(color='darkblue', width=2)
    )
))

fig.show()
```

### 5. Multi-Series & Grouping

#### Multiple Series

```python
import plotly.express as px

# Separate line per group
fig = px.line(
    df,
    x="date",
    y="value",
    color="category",  # Separate line per category
    title="Sales Over Time"
)
fig.show()
```

#### Grouped Bar Chart

```python
fig = px.bar(
    df,
    x="month",
    y="revenue",
    color="product",
    barmode="group",  # 'group' or 'stack'
    title="Revenue by Product"
)
fig.show()
```

#### Stacked Bar Chart

```python
fig = px.bar(
    df,
    x="month",
    y="revenue",
    color="product",
    barmode="stack",
    title="Total Revenue (Stacked)"
)
fig.show()
```

### 6. 3D Charts

#### 3D Scatter

```python
import plotly.express as px

fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    color="category",
    title="3D Scatter Plot"
)
fig.show()
```

#### 3D Surface

```python
import plotly.graph_objects as go
import numpy as np

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.update_layout(title="Surface Plot")
fig.show()
```

### 7. Subplots

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4")
)

# Add traces to specific subplots
fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="Series 1"), row=1, col=1)
fig.add_trace(go.Bar(x=[1, 2], y=[5, 6], name="Series 2"), row=1, col=2)
fig.add_trace(go.Scatter(x=[1, 2], y=[7, 8], name="Series 3"), row=2, col=1)
fig.add_trace(go.Scatter(x=[1, 2], y=[9, 10], name="Series 4"), row=2, col=2)

fig.update_layout(height=800, showlegend=False)
fig.show()
```

### 8. Hover & Click Events

#### Custom Hover Text

```python
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3],
    y=[4, 5, 6],
    hovertemplate='<b>Point %{x}</b><br>Value: %{y}<br><extra></extra>'
))

fig.update_layout(hovermode='x unified')
fig.show()
```

#### Annotations

```python
fig.add_annotation(
    x=2, y=5,
    text="Important Point",
    showarrow=True,
    arrowhead=2,
    ax=-40,
    ay=-40
)
fig.show()
```

## Common Patterns

### Pattern: Dashboard with Multiple Charts

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# Create subplots
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "bar"}, {"type": "scatter"}],
           [{"type": "histogram"}, {"type": "box"}]],
    subplot_titles=("Sales", "Trend", "Distribution", "Boxplot")
)

# Add different chart types
fig.add_trace(go.Bar(x=categories, y=sales), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=values, mode='lines'), row=1, col=2)
fig.add_trace(go.Histogram(x=data, nbinsx=20), row=2, col=1)
fig.add_trace(go.Box(y=values, name="Data"), row=2, col=2)

fig.update_layout(height=800, showlegend=False)
fig.show()
```

### Pattern: Interactive Dropdown Selection

```python
import plotly.graph_objects as go

# Data for different views
fig = go.Figure()

# Add traces for each view
for category in categories:
    data = df[df['category'] == category]
    fig.add_trace(go.Scatter(
        x=data['x'],
        y=data['y'],
        name=category,
        visible=(category == categories[0])  # Show first by default
    ))

# Create dropdown buttons
buttons = [
    dict(
        label="All",
        method="update",
        args=[{"visible": [True] * len(categories)},
              {"title": "All Categories"}]
    )
]

for i, category in enumerate(categories):
    visible = [j == i for j in range(len(categories))]
    buttons.append(
        dict(
            label=category,
            method="update",
            args=[{"visible": visible},
                  {"title": f"Category: {category}"}]
        )
    )

fig.update_layout(
    updatemenus=[dict(buttons=buttons, direction="down")],
    title="Select Category"
)
fig.show()
```

### Pattern: Time Series with Range Slider

```python
import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dates,
    y=prices,
    mode='lines',
    name='Price'
))

fig.update_layout(
    title="Stock Price with Range Slider",
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    ),
    yaxis_title="Price",
    hovermode='x unified'
)
fig.show()
```

## Best Practices

### ✅ DO: Use Plotly Express for Simple Charts

```python
# GOOD - Clean, concise syntax
fig = px.scatter(df, x="x", y="y", color="category")
fig.show()
```

### ❌ DON'T: Always Use Graph Objects

```python
# BAD - Verbose for simple charts
fig = go.Figure(data=[go.Scatter(x=df['x'], y=df['y'])])
```

### ✅ DO: Name Your Traces

```python
# GOOD - Clear legend
fig.add_trace(go.Scatter(..., name="Revenue"))
fig.add_trace(go.Scatter(..., name="Expenses"))
```

### ✅ DO: Use .show() in Jupyter, Save for Deployment

```python
# For notebooks
fig.show()

# For web/export
fig.write_html("chart.html")
fig.write_image("chart.png")  # Requires kaleido
```

### ✅ DO: Set Appropriate Hover Information

```python
# GOOD - Helpful hover text
fig = px.scatter(
    df, x="x", y="y",
    hover_data=["category", "value"]
)
```

### ❌ DON'T: Overcrowd Charts

```python
# BAD - Too much data at once
fig = px.scatter(df, x="x", y="y", color="cat1", size="s1",
                 hover_data=[col for col in df.columns])

# GOOD - Focus on key information
fig = px.scatter(df, x="x", y="y", color="category")
```

### ✅ DO: Use Templates for Consistent Styling

```python
# GOOD - Apply theme to all charts
fig.update_layout(template="plotly_white")
```

## Common Issues & Solutions

### Issue: Chart Not Displaying

**Problem**: `fig.show()` doesn't display chart in notebook.

**Solution**: Ensure Jupyter environment or use `fig.write_html()` to save.

```python
# In Jupyter
fig.show()

# Save as HTML
fig.write_html("chart.html")
```

### Issue: Hover Text Not Working

**Problem**: Hover information not displaying.

**Solution**: Use `.hover_data` parameter.

```python
fig = px.scatter(
    df, x="x", y="y",
    hover_data=["value", "category"]  # Add columns to show
)
```

### Issue: Legend Too Large

**Problem**: Legend overlaps chart.

**Solution**: Position legend outside chart area.

```python
fig.update_layout(
    legend=dict(
        x=1.02,  # Outside plot area
        y=1,
        bgcolor='rgba(255, 255, 255, 0.8)',
        border=dict(color='black', width=1)
    )
)
```

### Issue: Exporting to PNG Requires kaleido

**Problem**: `write_image()` fails.

**Solution**: Install kaleido.

```bash
pip install kaleido
```

## API Reference - Quick Lookup

### Plotly Express Common Functions

```python
px.scatter()        # Scatter plot
px.line()           # Line chart
px.bar()            # Bar chart
px.histogram()      # Histogram
px.box()            # Box plot
px.violin()         # Violin plot
px.strip()          # Strip plot
px.scatter_matrix() # Scatter matrix
px.sunburst()       # Sunburst chart
px.treemap()        # Treemap
px.funnel()         # Funnel chart
px.scatter_3d()     # 3D scatter
px.line_3d()        # 3D line
```

### Common Parameters

```python
px.scatter(
    data_frame=df,           # DataFrame
    x="column",              # X axis
    y="column",              # Y axis
    color="column",          # Color by column
    size="column",           # Size by column
    hover_data=["col1"],     # Hover info
    title="Title",           # Chart title
    labels={"x": "X Label"}  # Custom labels
)
```

### Graph Objects Traces

```python
go.Scatter()        # Scatter/line
go.Bar()            # Bar chart
go.Histogram()      # Histogram
go.Box()            # Box plot
go.Violin()         # Violin plot
go.Heatmap()        # Heatmap
go.Contour()        # Contour plot
go.Surface()        # 3D surface
go.Mesh3d()         # 3D mesh
go.Scatter3d()      # 3D scatter
go.Sunburst()       # Sunburst
go.Treemap()        # Treemap
```

## Export Options

```python
# Display in notebook/browser
fig.show()

# Save as HTML (standalone file)
fig.write_html("chart.html")

# Save as static image (requires kaleido)
fig.write_image("chart.png")
fig.write_image("chart.pdf")
fig.write_image("chart.svg")

# Get JSON representation
fig.to_json()

# Get dict representation
fig.to_dict()
```

## Additional Resources

- **Official Docs**: https://plotly.com/python/
- **GitHub**: https://github.com/plotly/plotly.py
- **Chart Gallery**: https://plotly.com/python/
- **Dash Framework**: https://dash.plotly.com/ (interactive web apps)
- **Community Discourse**: https://community.plotly.com/

## Integration with Other Libraries

### With Pandas DataFrames

```python
import plotly.express as px
import pandas as pd

df = pd.read_csv("data.csv")
fig = px.scatter(df, x="column_x", y="column_y", color="category")
fig.show()
```

### With Polars DataFrames

```python
import plotly.express as px
import polars as pl

df = pl.read_csv("data.csv")
fig = px.scatter(df.to_pandas(), x="column_x", y="column_y")
fig.show()
```

### With Marimo Notebooks

```python
import marimo as mo
import plotly.express as px

fig = px.scatter(df, x="x", y="y")
mo.Html(fig.to_html())  # Display in marimo
```

### With Dash for Interactive Apps

```python
from dash import Dash, dcc, html
import plotly.express as px

app = Dash(__name__)

fig = px.scatter(df, x="x", y="y")

app.layout = html.Div([
    html.H1("Dashboard"),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)
```
