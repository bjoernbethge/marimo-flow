import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""Openvino snippets""")
    return


@app.cell
def _():
    from openvino_genai import LLMPipeline, GenerationConfig
    return GenerationConfig, LLMPipeline


@app.cell
def _(GenerationConfig):
    config = GenerationConfig()
    config.max_new_tokens = 100
    config.temperature = 0.7
    return (config,)


@app.cell
def _(config, mo):
    get_temperature, set_temperature = mo.state(config.temperature)
    mo.ui.slider(
        start=0.1,
        stop=1.0,
        step=0.01,
        value=get_temperature(),
        on_change=set_temperature,
        show_value=True,
    )
    return


@app.cell
def _(LLMPipeline, device, kwargs, model, tokenizer, weights):
    pipe = LLMPipeline(model, weights, tokenizer, device, kwargs)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
