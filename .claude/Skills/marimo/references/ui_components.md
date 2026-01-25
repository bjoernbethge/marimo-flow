# UI Components Reference

Complete catalog of marimo UI elements and layout components.

## Numeric Inputs

### Slider
```python
slider = mo.ui.slider(
    start=0,
    stop=100,
    value=50,
    label="Threshold",
    step=1,
    debounce=False
)
```

### Number Input
```python
number = mo.ui.number(
    start=0,
    stop=150,
    step=1,
    value=25,
    label="Age"
)
```

### Range Slider
```python
range_slider = mo.ui.range_slider(
    start=0,
    stop=100,
    value=[25, 75],
    label="Range"
)
```

## Text Inputs

### Text Field
```python
text = mo.ui.text(
    value="",
    placeholder="Enter name",
    label="Name",
    kind="text"  # "text", "email", "password", "url"
)
```

### Text Area
```python
text_area = mo.ui.text_area(
    value="",
    placeholder="Description",
    label="Description",
    rows=5
)
```

## Selection Inputs

### Dropdown
```python
dropdown = mo.ui.dropdown(
    options=["Option 1", "Option 2", "Option 3"],
    value="Option 1",
    label="Select",
    allow_select_none=False
)
```

### Radio Buttons
```python
radio = mo.ui.radio(
    options=["A", "B", "C"],
    value="A",
    label="Choose"
)
```

### Checkbox
```python
checkbox = mo.ui.checkbox(
    value=False,
    label="I agree"
)
```

### Multi-Select
```python
multiselect = mo.ui.multiselect(
    options=["Red", "Green", "Blue"],
    value=["Red"],
    label="Select colors"
)
```

## Action Buttons

### Standard Button
```python
button = mo.ui.button(
    label="Click me",
    on_change=lambda v: print(v),
    kind="neutral"  # "neutral", "success", "warn", "danger"
)
```

### Run Button
```python
run_button = mo.ui.run_button(
    label="Execute",
    kind='primary'
)
```

## Data Display

### Table
```python
table = mo.ui.table(
    data,
    sortable=True,
    filterable=True,
    selection="multi",  # "single", "multi", None
    pagination=True,
    page_size=10
)
```

### Data Explorer
```python
data_explorer = mo.ui.data_explorer(df)
```

### DataFrame Display
```python
dataframe = mo.ui.dataframe(
    df,
    selection="multi",
    page_size=10
)
```

## Charts (Reactive)

### Altair Chart
```python
altair_chart = mo.ui.altair_chart(
    chart,
    chart_selection=alt.selection_interval()
)
# Access selected data
selected = altair_chart.value
```

### Plotly Chart
```python
plotly_chart = mo.ui.plotly(
    fig,
    config={"displayModeBar": True}
)
```

## File Upload

### File Input
```python
file = mo.ui.file(
    label='Upload file',
    multiple=False,
    kind="button",  # "button", "area"
    filetypes=[".csv", ".txt"]
)

# Access file contents
if file.value:
    content = file.value[0].contents  # bytes
    name = file.value[0].name
```

## Layout Components

### Vertical Stack
```python
sidebar = mo.vstack(
    [
        mo.md("## Controls"),
        slider,
        dropdown,
        button
    ],
    gap=1.5,
    align="stretch"  # "start", "center", "end", "stretch"
)
```

### Horizontal Stack
```python
metrics = mo.hstack(
    [
        mo.stat(label="Users", value=1234),
        mo.stat(label="Revenue", value="$56.7K")
    ],
    gap=2,
    justify="space-between",  # "start", "center", "end", "space-between"
    widths="equal"  # "equal" or list of widths
)
```

### Tabs
```python
tabs = mo.tabs({
    "Overview": mo.md("Dashboard overview"),
    "Analytics": chart,
    "Settings": settings_form
})
```

### Accordion
```python
accordion = mo.accordion({
    "Section 1": content1,
    "Section 2": content2
})
```

### Center Content
```python
mo.center(mo.md("# Welcome"))
```

### Callout Box
```python
mo.callout(
    mo.md("Important information"),
    kind="info"  # "neutral", "info", "warn", "success", "danger"
)
```

### Statistics Display
```python
mo.stat(
    label="Total Users",
    value=1234,
    caption="Active users",
    direction="increase"  # "increase", "decrease", None
)
```

## Forms and Gated Execution

### Form Wrapper
```python
# Wrap any UI element in a form
form = mo.ui.text(label="Your name").form()
form  # Display

# Value only available after submission
form.value  # None until submitted
```

### Batch Updates with mo.batch
```python
# Update multiple states atomically
with mo.batch():
    set_state_1(value1)
    set_state_2(value2)
    set_state_3(value3)
# UI updates once after all state changes
```

## Accessing UI Values

**Critical Rule**: Access `.value` in a DIFFERENT cell than where element is defined.

```python
# Cell 1: Define UI element
slider = mo.ui.slider(0, 100, value=50)
slider  # Display it

# Cell 2: Use its value
result = slider.value * 2
mo.md(f"Double the value: {result}")
```

## Dynamic Collections

### Array of Elements
```python
sliders = mo.ui.array([
    mo.ui.slider(0, 100, label=f"Slider {i}")
    for i in range(5)
])

# Access values
values = sliders.value  # Returns list of values
```

### Dictionary of Elements
```python
form_elements = mo.ui.dictionary({
    "First name": mo.ui.text(placeholder="First name"),
    "Last name": mo.ui.text(placeholder="Last name"),
    "Email": mo.ui.text(placeholder="Email", kind="email")
})

# Access values
data = form_elements.value  # Returns dict of values
```

## Embedding UI in Markdown

```python
elements = mo.ui.dictionary({
    "checks": mo.ui.array([mo.ui.checkbox() for _ in range(3)]),
    "texts": mo.ui.array([mo.ui.text() for _ in range(3)])
})

mo.md(f"""
# Task List
{chr(10).join([
    f"{check} {text}"
    for check, text in zip(elements["checks"], elements["texts"])
])}
""")
```
