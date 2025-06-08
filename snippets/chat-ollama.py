import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Chat with Ollama""")
    return

@app.cell
def _():
    from openai import OpenAI
    return


@app.cell
def _(mo):
    chat = mo.ui.chat(
        mo.ai.llm.openai(
            "hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M",
            base_url="http://localhost:11434/v1",
            system_message="You are a helpful assistant.",
        ),
        prompts=["Write a haiku about recursion in programming."],
    )
    chat
    return

@app.cell
def _():
    import marimo as mo
    return (mo,)

if __name__ == "__main__":
    app.run()
