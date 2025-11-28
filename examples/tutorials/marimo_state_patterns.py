"""
Marimo State Management Patterns

Based on docs/marimo-quickstart.md State Patterns
Demonstrates:
- Counter pattern with mo.state()
- Todo list state management
- Shopping cart pattern
- Undo/redo functionality
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
        # Marimo State Management Patterns

        **State** enables interactive applications with `mo.state()`.

        ## Patterns
        - 🔢 Counter (increment/decrement)
        - ✅ Todo List (add/remove/complete)
        - 🛒 Shopping Cart (add/remove items)
        - ↩️ Undo/Redo stack
        """
    )
    return


@app.cell
def _(mo):
    """Pattern selector"""
    pattern = mo.ui.dropdown(
        options=["Counter", "Todo List", "Shopping Cart", "Undo/Redo"],
        value="Counter",
        label="📋 State Pattern",
    )
    mo.md(f"## Select Pattern\n\n{pattern}")
    return (pattern,)


@app.cell
def _(mo, pattern):
    """Implement selected pattern"""

    if pattern.value == "Counter":
        # Counter pattern
        count, set_count = mo.state(0)

        increment = mo.ui.button(label="+1", on_click=lambda _: set_count(count + 1))
        decrement = mo.ui.button(label="-1", on_click=lambda _: set_count(count - 1))
        reset = mo.ui.button(label="Reset", on_click=lambda _: set_count(0))

        output = mo.md(
            f"""
            ### Counter: {count}

            {increment} {decrement} {reset}
            """
        )

    elif pattern.value == "Todo List":
        # Todo list pattern
        todos, set_todos = mo.state([])
        new_todo = mo.ui.text(placeholder="Add todo...")

        def add_todo():
            if new_todo.value:
                set_todos(todos + [{"text": new_todo.value, "done": False}])

        add_btn = mo.ui.button(label="Add", on_click=lambda _: add_todo())

        todos_html = "<br>".join(
            [
                f"{'✅' if t['done'] else '⬜'} {t['text']}"
                for t in todos
            ]
        ) or "_No todos yet_"

        output = mo.md(
            f"""
            ### Todo List

            {new_todo} {add_btn}

            {todos_html}

            **Total:** {len(todos)} todos ({sum(1 for t in todos if t['done'])} done)
            """
        )

    elif pattern.value == "Shopping Cart":
        # Shopping cart pattern
        cart, set_cart = mo.state([])

        products = [
            {"id": 1, "name": "Laptop", "price": 999},
            {"id": 2, "name": "Mouse", "price": 25},
            {"id": 3, "name": "Keyboard", "price": 75},
        ]

        def add_to_cart(product):
            set_cart(cart + [product])

        cart_html = "<br>".join(
            [f"{item['name']} - ${item['price']}" for item in cart]
        ) or "_Empty cart_"

        total = sum(item["price"] for item in cart)

        output = mo.md(
            f"""
            ### Shopping Cart

            **Products:**
            {" ".join([f"`{p['name']}` " for p in products])}

            **Cart:**
            {cart_html}

            **Total:** ${total}
            """
        )

    else:  # Undo/Redo
        # Undo/redo pattern
        history, set_history = mo.state([0])
        pos, set_pos = mo.state(0)

        current = history[pos] if history else 0

        def add_value(delta):
            new_val = current + delta
            new_history = history[: pos + 1] + [new_val]
            set_history(new_history)
            set_pos(pos + 1)

        undo = mo.ui.button(
            label="Undo", on_click=lambda _: set_pos(max(0, pos - 1)), disabled=pos == 0
        )
        redo = mo.ui.button(
            label="Redo",
            on_click=lambda _: set_pos(min(len(history) - 1, pos + 1)),
            disabled=pos >= len(history) - 1,
        )

        plus = mo.ui.button(label="+1", on_click=lambda _: add_value(1))
        minus = mo.ui.button(label="-1", on_click=lambda _: add_value(-1))

        output = mo.md(
            f"""
            ### Undo/Redo Counter

            **Value:** {current}

            {plus} {minus}

            {undo} {redo}

            _History: {' → '.join(map(str, history))}_
            """
        )

    return (output,)


@app.cell
def _(mo, output):
    """Display pattern"""
    output
    return


@app.cell
def _(mo):
    """Code examples"""
    mo.md(
        r"""
        ## 💻 State Patterns

        ### Counter
        ```python
        count, set_count = mo.state(0)
        increment = mo.ui.button(on_click=lambda _: set_count(count + 1))
        ```

        ### Todo List
        ```python
        todos, set_todos = mo.state([])
        set_todos(todos + [new_item])  # Add
        set_todos([t for t in todos if t != item])  # Remove
        ```

        ### Undo/Redo
        ```python
        history, set_history = mo.state([initial])
        position, set_position = mo.state(0)
        # Add to history on change
        # Move position for undo/redo
        ```

        ## 📚 Learn More
        - See `docs/marimo-quickstart.md` for state patterns
        """
    )
    return


if __name__ == "__main__":
    app.run()
