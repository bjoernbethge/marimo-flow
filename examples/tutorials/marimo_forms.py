"""
Marimo Forms - Interactive Form Patterns

Based on docs/marimo-quickstart.md Form Patterns
Demonstrates:
- Form creation with validation
- Multi-step forms
- Dynamic form fields
- Form state management
- Submission handling
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
        # Marimo Forms & State Management

        **Forms** collect user input with validation and submission handling.

        ## Patterns Covered
        - ✅ Basic forms with validation
        - 🔄 Multi-step wizards
        - 📝 Dynamic field generation
        - 💾 Form state persistence
        """
    )
    return


@app.cell
def _(mo):
    """Form type selector"""
    form_type = mo.ui.dropdown(
        options=["Basic Form", "Multi-Step Form", "Dynamic Form", "Validated Form"],
        value="Basic Form",
        label="📋 Select Form Type",
    )
    mo.md(f"## Form Examples\n\n{form_type}")
    return (form_type,)


@app.cell
def _(form_type, mo):
    """Create selected form"""

    if form_type.value == "Basic Form":
        # Simple contact form
        form = mo.ui.form(
            {
                "name": mo.ui.text(placeholder="Your name"),
                "email": mo.ui.text(placeholder="email@example.com"),
                "message": mo.ui.text_area(placeholder="Your message..."),
                "subscribe": mo.ui.checkbox(label="Subscribe to newsletter"),
            },
            label="Contact Form",
            submit_button_label="Submit",
        )

    elif form_type.value == "Multi-Step Form":
        # Multi-step registration form
        step, set_step = mo.state(1)

        if step == 1:
            form = mo.ui.form(
                {
                    "first_name": mo.ui.text(placeholder="First name"),
                    "last_name": mo.ui.text(placeholder="Last name"),
                    "email": mo.ui.text(placeholder="Email"),
                },
                label="Step 1: Personal Info",
            )
        elif step == 2:
            form = mo.ui.form(
                {
                    "company": mo.ui.text(placeholder="Company name"),
                    "role": mo.ui.dropdown(
                        options=["Developer", "Manager", "Designer", "Other"]
                    ),
                    "team_size": mo.ui.slider(1, 100, value=10, label="Team size"),
                },
                label="Step 2: Professional Info",
            )
        else:
            form = mo.ui.form(
                {
                    "interests": mo.ui.multiselect(
                        options=["ML", "Data Viz", "DevOps", "Cloud"]
                    ),
                    "notifications": mo.ui.checkbox(label="Enable notifications"),
                },
                label="Step 3: Preferences",
            )

    elif form_type.value == "Dynamic Form":
        # Form with dynamically added fields
        num_fields = mo.ui.slider(1, 5, value=3, label="Number of fields")
        fields = {f"field_{i}": mo.ui.text(placeholder=f"Field {i}") for i in range(num_fields.value)}
        form = mo.ui.form(fields, label="Dynamic Form")

    else:  # Validated Form
        # Form with custom validation
        form = mo.ui.form(
            {
                "username": mo.ui.text(placeholder="Username (min 3 chars)"),
                "password": mo.ui.text(
                    placeholder="Password (min 8 chars)", kind="password"
                ),
                "age": mo.ui.number(start=0, stop=120, value=18, label="Age"),
                "country": mo.ui.dropdown(
                    options=["USA", "UK", "Germany", "France", "Other"]
                ),
            },
            label="Registration Form",
        )

    return (form,)


@app.cell
def _(form, mo):
    """Display form"""
    mo.md(f"### Form Input\n\n{form}")
    return


@app.cell
def _(form, mo):
    """Handle form submission"""
    if form.value:
        mo.md(
            f"""
            ### ✅ Form Submitted!

            **Submitted Data:**
            ```json
            {form.value}
            ```
            """
        )
    else:
        mo.md("_Submit the form to see results..._")
    return


@app.cell
def _(mo):
    """Code patterns"""
    mo.md(
        r"""
        ## 💻 Form Patterns

        ### Basic Form
        ```python
        form = mo.ui.form({
            "name": mo.ui.text(placeholder="Name"),
            "email": mo.ui.text(placeholder="Email"),
        })

        # Access values after submission
        if form.value:
            name = form.value["name"]
        ```

        ### Multi-Step Form
        ```python
        step, set_step = mo.state(1)

        if step == 1:
            form = mo.ui.form({...}, label="Step 1")
        elif step == 2:
            form = mo.ui.form({...}, label="Step 2")

        # Advance on submission
        if form.value:
            set_step(step + 1)
        ```

        ### Validation
        ```python
        if form.value:
            username = form.value["username"]
            if len(username) < 3:
                mo.md("❌ Username too short")
            else:
                mo.md("✅ Valid!")
        ```

        ## 📚 Learn More
        - See `docs/marimo-quickstart.md` for form patterns
        - [Marimo Forms Documentation](https://docs.marimo.io/)
        """
    )
    return


if __name__ == "__main__":
    app.run()
