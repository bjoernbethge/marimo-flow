"""Task Manager Application

Demonstrates state management with a task tracking application:
- mo.state() for managing task list
- Button callbacks for state updates
- Dynamic UI collections
- Form-like interactions
"""

import marimo

app = marimo.App(width="medium", app_title="Task Manager")


@app.cell
def imports():
    """Import required libraries"""
    import marimo as mo
    from dataclasses import dataclass
    from datetime import datetime
    return mo, dataclass, datetime


@app.cell
def task_model(dataclass):
    """Define task data model"""
    @dataclass
    class Task:
        id: int
        name: str
        done: bool = False
        created_at: str = ""

        def __post_init__(self):
            if not self.created_at:
                from datetime import datetime
                self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return Task,


@app.cell
def task_state(mo):
    """Initialize task state"""
    # State for task list
    get_tasks, set_tasks = mo.state([])

    # State for next task ID
    get_next_id, set_next_id = mo.state(1)

    return get_tasks, set_tasks, get_next_id, set_next_id,


@app.cell
def task_input(mo, get_tasks, set_tasks, get_next_id, set_next_id, Task):
    """UI for adding new tasks"""
    # Input field
    task_entry = mo.ui.text(
        placeholder="Enter task description...",
        label="New Task"
    )

    # Add button with callback
    def add_task(_):
        if task_entry.value.strip():
            new_task = Task(
                id=get_next_id(),
                name=task_entry.value.strip()
            )
            set_tasks(lambda tasks: tasks + [new_task])
            set_next_id(lambda id: id + 1)

    add_button = mo.ui.button(
        label="Add Task",
        on_change=add_task,
        kind="success"
    )

    # Layout
    input_section = mo.hstack([
        task_entry,
        add_button
    ], gap=1, justify="start")

    input_section  # Display

    return task_entry, add_button,


@app.cell
def task_actions(mo, get_tasks, set_tasks):
    """Buttons for task list actions"""
    # Clear completed tasks
    clear_button = mo.ui.button(
        label="Clear Completed",
        on_change=lambda _: set_tasks(
            lambda tasks: [t for t in tasks if not t.done]
        ),
        kind="warn"
    )

    # Clear all tasks
    clear_all_button = mo.ui.button(
        label="Clear All",
        on_change=lambda _: set_tasks([]),
        kind="danger"
    )

    # Layout
    actions = mo.hstack([
        clear_button,
        clear_all_button
    ], gap=1)

    actions  # Display

    return clear_button, clear_all_button,


@app.cell
def task_list_display(mo, get_tasks, set_tasks):
    """Display task list with checkboxes"""
    tasks = get_tasks()

    if not tasks:
        mo.md("*No tasks yet. Add one above!*")
    else:
        # Create checkbox for each task
        task_checkboxes = mo.ui.array(
            [
                mo.ui.checkbox(
                    value=task.done,
                    label=f"{task.name} (added {task.created_at})"
                )
                for task in tasks
            ],
            on_change=lambda values: set_tasks(
                lambda tasks: [
                    Task(
                        id=t.id,
                        name=t.name,
                        done=values[i],
                        created_at=t.created_at
                    )
                    for i, t in enumerate(tasks)
                ]
            )
        )

        # Display task list
        mo.vstack([
            mo.md("## Task List"),
            task_checkboxes
        ], gap=1)

    return


@app.cell
def task_statistics(mo, get_tasks):
    """Display task statistics"""
    tasks = get_tasks()

    total = len(tasks)
    completed = sum(1 for t in tasks if t.done)
    pending = total - completed
    completion_rate = (completed / total * 100) if total > 0 else 0

    stats = mo.hstack([
        mo.stat(
            label="Total Tasks",
            value=total,
            caption="All tasks"
        ),
        mo.stat(
            label="Completed",
            value=completed,
            caption=f"{completion_rate:.0f}% done",
            direction="increase" if completion_rate > 50 else None
        ),
        mo.stat(
            label="Pending",
            value=pending,
            caption="To do"
        )
    ], gap=2, widths="equal")

    stats  # Display

    return


@app.cell
def usage_notes(mo):
    """Display usage instructions"""
    mo.accordion({
        "Usage Instructions": mo.md("""
        ### How to use this Task Manager

        1. **Add Task**: Enter task description and click "Add Task"
        2. **Mark Complete**: Check the box next to completed tasks
        3. **Clear Completed**: Remove all completed tasks from the list
        4. **Clear All**: Remove all tasks (use with caution!)

        Tasks are stored in memory and will reset when you restart the notebook.
        """)
    })
    return


if __name__ == "__main__":
    app.run()
