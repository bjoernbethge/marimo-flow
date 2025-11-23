# /// script
# dependencies = [
#     "marimo",
#     "ollama>=0.3.0",
#     "pydantic>=2.10.0",
#     "pydantic-ai>=0.0.6",
#     "pydantic-ai-slim[mcp]>=0.0.6",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# ///

import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import ollama
    from pydantic import BaseModel
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.mcp import load_mcp_servers
    return Agent, BaseModel, OpenAIChatModel, OpenAIProvider, mo, ollama


@app.cell
def _(mo, ollama):
    get_model_list, set_model_list  = mo.state([model.model for model in ollama.list().models])
    get_model, set_model = mo.state(get_model_list()[0])

    return get_model, get_model_list, set_model, set_model_list


@app.cell
def _(mo):
    tabs = mo.ui.tabs({
        "🤖 Models": "models",
        "💬 Chat": "chat",
        "🧬 Embeddings": "embeddings",
        "⚙️ System": "system"
    })
    
    mo.md(f"""
    # 🦙 Ollama Manager
    
    {tabs}
    """)
    return (tabs,)


@app.function
def pull_model(model_url, get_model_list, set_model_list, set_model, ollama):
    """Pull a model from ollama and update the model list"""
    ollama.pull(model_url)
    set_model_list([model.model for model in ollama.list().models])
    set_model(get_model_list()[0])


@app.cell
def _(get_model_list, mo, model_url, ollama, set_model, set_model_list):
    pull_button = mo.ui.button(
        label='⬇️', 
        value=model_url.value, 
        kind="warn", 
        on_change=lambda url: pull_model(url, get_model_list, set_model_list, set_model, ollama)
    )
    return mo.vstack([model_url, pull_button], justify="start")


@app.function
def delete_model(model_id, get_model_list, set_model_list, set_model, ollama):
    """Delete a model from ollama and update the model list"""
    ollama.delete(model_id)
    set_model_list([model.model for model in ollama.list().models])
    set_model(get_model_list()[0])


@app.cell
def _(get_model, get_model_list, mo, ollama, set_model, set_model_list):
    delete_button = mo.ui.button(
        label='❌', 
        value=get_model(), 
        kind="danger", 
        on_change=lambda model_id: delete_model(model_id, get_model_list, set_model_list, set_model, ollama)
    )
    model = mo.ui.dropdown(
        options=get_model_list(),
        label="Model",
        value=get_model(),
        on_change=set_model
    )
    return delete_button, model


@app.function
def show_model_info(get_model, ollama, mo):
    """Display model information as markdown"""
    model_info = ollama.show(get_model()).details
    if model_info.parent_model == '':
        model_info.parent_model = get_model()
    return mo.md(
        rf"""

    🧠{model_info.parent_model}
    👨‍👩‍👧{model_info.family}
    💽{model_info.format}
    ↔️{model_info.parameter_size}
    🦑{model_info.quantization_level}
    """
    )


@app.cell
def _(get_model, ollama):
    model_data = ollama.show(get_model())
    modelinfo = model_data.modelinfo
    template = model_data.template
    return modelinfo, template


@app.cell
def _(
    delete_button,
    get_model,
    mo,
    model,
    model_url,
    modelinfo,
    ollama,
    pull_button,
    show_model_info,
    tabs,
    template,
):
    if tabs.value == "models":
        mo.vstack(
            [
                mo.md("## 📥 Pull Model"),
                mo.hstack([model_url, pull_button], justify="start"),
                mo.md("## 🛠️ Manage Models"),
                mo.hstack([model, delete_button], justify="start"),
                show_model_info(get_model, ollama, mo),
                mo.accordion(
                    {
                        "More Info": mo.accordion(
                            {
                                "Details": mo.md(rf"{modelinfo}"),
                                "Template": mo.md(f"```\n{template}\n```"),
                            }
                        )
                    }
                ),
            ]
        )
    return


@app.cell
async def _(Agent, BaseModel, OpenAIChatModel, OpenAIProvider, get_model):



    class CityLocation(BaseModel):
        city: str
        country: str


    ollama_model = OpenAIChatModel(
        model_name=get_model(), provider=OpenAIProvider(base_url='http://localhost:11434/v1')
    )
    agent = Agent(ollama_model, output_type=CityLocation)

    result = await agent.run('Where were the olympics held in 2012?')
    print(result)
    return


@app.function
def ollama_chat(messages, get_model, ollama):
    """Chat with ollama model"""
    try:
        response = ollama.chat(
            model=get_model(),
            messages=messages
        )
        print(response)
        return response
    except ollama.ResponseError as e:
        error_message = f'Error: {e.error}'
        print(error_message)
        return error_message


@app.cell
def _():
    return


@app.cell
def _(get_model, mo):
    chat = mo.ui.chat(
        mo.ai.llm.openai(
            get_model(),
            api_key='ollama',
            # Change this if you are using a different OpenAI-compatible endpoint.
            base_url="http://localhost:11434/v1",
            system_message="You are a helpful assistant.",
        ),
        prompts=["Write a haiku about recursion in programming."],
    )
    chat
    return


@app.cell
def _(mo, model):


    form = (
        mo.md(
            '''
        {name}
        {parent_model}

        **System prompt.**

        {system_prompt}

        '''
        )
        .batch(
            name=mo.ui.text(label="Name"),
            parent_model=model,
            system_prompt=mo.ui.text_area(),
        )
        .form()
    )
    form
    return (form,)


@app.cell
def _(form):
    form.value
    return


@app.cell
def _(form, mo, ollama):
    mo.stop(form.value is None, mo.md("Fill model form"))
    m_d = form.value
    ollama.create(model=m_d["name"], from_=m_d["parent_model"], system=m_d["system_prompt"])
    return


@app.cell
def _(mo):
    # Create a button to copy a model
    copy_button = mo.ui.run_button(label='Copy Model')

    # Display the button
    copy_button
    return (copy_button,)


@app.cell
def _(copy_button, ollama):
    # Copy a model when button is pressed
    if copy_button.value:
        copied_model = ollama.copy('llama3.2', 'user/llama3.2')
        copied_model
    return


@app.cell
def _(mo):
    mo.md(r"""👟""")
    return


@app.cell
def _(mo):
    # Create a button to push a model
    push_button = mo.ui.run_button(label='Push Model')

    # Display the button
    push_button
    return (push_button,)


@app.cell
def _(ollama, push_button):
    # Push a model when button is pressed
    if push_button.value:
        pushed_model = ollama.push('user/llama3.2')
        pushed_model
    return


@app.cell
def _(mo):
    # Create a text area for embedding input
    embed_input = mo.ui.text_area(value='', label='Embed Input')

    # Create a button to embed input
    embed_button = mo.ui.run_button(label='Embed Input')

    # Display UI elements
    mo.vstack([embed_input, embed_button])
    return embed_button, embed_input


@app.cell
def _(embed_button, embed_input, get_model, ollama):
    # Embed input when button is pressed
    if embed_button.value:
        embedding = ollama.embed(model=get_model(), input=embed_input.value)
        embedding
    return (embedding,)


@app.cell
def _(embedding):
    embedding
    return


@app.cell
def _(mo):
    # Create a button to embed input in batch
    embed_batch_button = mo.ui.run_button(label='Embed Input (Batch)')

    # Display the button
    embed_batch_button
    return (embed_batch_button,)


@app.cell
def _(embed_batch_button, embed_input, get_model, ollama):
    # Embed input in batch when button is pressed
    if embed_batch_button.value:
        batch_embedding = ollama.embed(model=get_model(), input=[embed_input.value, 'Grass is green because of chlorophyll'])
        batch_embedding
    return


@app.cell
def _(mo):
    # Create a button to list running processes
    ps_button = mo.ui.run_button(label='List Running Processes')

    # Display the button
    ps_button
    return (ps_button,)


@app.cell
def _(ollama, ps_button):
    # List running processes when button is pressed
    if ps_button.value:
        processes = ollama.ps()
        print(processes)
    return


if __name__ == "__main__":
    app.run()
