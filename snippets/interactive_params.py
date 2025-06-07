import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Interactive Parameter Controls""")
    return

@app.cell
def _(mo):
    """Numeric Parameters"""
    learning_rate = mo.ui.slider(
        start=0.001, stop=0.1, step=0.001, value=0.01,
        label="📈 Learning Rate"
    )
    
    batch_size = mo.ui.dropdown(
        options=[16, 32, 64, 128, 256],
        value=32,
        label="📦 Batch Size"
    )
    
    epochs = mo.ui.number(
        start=1, stop=1000, step=1, value=100,
        label="🔄 Epochs"
    )
    
    mo.md(f"""
    ## 🎯 Training Parameters
    {learning_rate}
    {batch_size}
    {epochs}
    """)
    return batch_size, epochs, learning_rate

@app.cell
def _(mo):
    """Model Architecture"""
    model_architecture = mo.ui.dropdown(
        options=["Simple", "Deep", "Wide", "ResNet-like"],
        value="Simple",
        label="🏗️ Architecture"
    )
    
    activation = mo.ui.dropdown(
        options=["relu", "tanh", "sigmoid", "gelu"],
        value="relu",
        label="⚡ Activation Function"
    )
    
    dropout_rate = mo.ui.slider(
        start=0.0, stop=0.8, step=0.1, value=0.2,
        label="🎭 Dropout Rate"
    )
    
    mo.md(f"""
    ## 🧠 Model Configuration
    {model_architecture}
    {activation}
    {dropout_rate}
    """)
    return activation, dropout_rate, model_architecture

@app.cell
def _(mo):
    """Data Processing"""
    normalize_data = mo.ui.checkbox(
        value=True,
        label="📊 Normalize Data"
    )
    
    feature_selection = mo.ui.multiselect(
        options=["PCA", "SelectKBest", "RFE", "Variance"],
        value=["SelectKBest"],
        label="🎯 Feature Selection"
    )
    
    train_test_split_ratio = mo.ui.slider(
        start=0.1, stop=0.5, step=0.05, value=0.2,
        label="✂️ Test Split Ratio"
    )
    
    mo.md(f"""
    ## 🔧 Data Processing
    {normalize_data}
    {feature_selection}
    {train_test_split_ratio}
    """)
    return feature_selection, normalize_data, train_test_split_ratio

@app.cell
def _(
    activation,
    batch_size,
    dropout_rate,
    epochs,
    feature_selection,
    learning_rate,
    model_architecture,
    mo,
    normalize_data,
    train_test_split_ratio,
):
    """Parameter Summary"""
    config = {
        "training": {
            "learning_rate": learning_rate.value,
            "batch_size": batch_size.value,
            "epochs": epochs.value
        },
        "model": {
            "architecture": model_architecture.value,
            "activation": activation.value,
            "dropout_rate": dropout_rate.value
        },
        "data": {
            "normalize": normalize_data.value,
            "feature_selection": feature_selection.value,
            "test_split": train_test_split_ratio.value
        }
    }
    
    mo.md(f"""
    ## 📋 Configuration Summary
    
    ### Training Parameters
    - **Learning Rate**: {config['training']['learning_rate']}
    - **Batch Size**: {config['training']['batch_size']}
    - **Epochs**: {config['training']['epochs']}
    
    ### Model Architecture
    - **Type**: {config['model']['architecture']}
    - **Activation**: {config['model']['activation']}
    - **Dropout**: {config['model']['dropout_rate']}
    
    ### Data Processing
    - **Normalize**: {config['data']['normalize']}
    - **Feature Selection**: {', '.join(config['data']['feature_selection'])}
    - **Test Split**: {config['data']['test_split']}
    
    **Ready for training!** ✅
    """)
    return (config,)

@app.cell
def _():
    import marimo as mo
    return (mo,)

if __name__ == "__main__":
    app.run() 