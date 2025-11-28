"""Callbacks for PINA training monitoring."""

from __future__ import annotations

from typing import Any

from lightning.pytorch.callbacks import Callback
import marimo as mo
import polars as pl
import altair as alt


class MarimoLivePlotter(Callback):
    """Callback to stream training metrics to a Marimo chart."""

    def __init__(self, update_every_n_epochs: int = 10) -> None:
        self.update_every_n_epochs = update_every_n_epochs
        self.history: list[dict[str, Any]] = []
        self.chart_container = mo.empty()
        self.table_container = mo.empty()

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Called when the train epoch ends."""
        epoch = trainer.current_epoch
        metrics = {
            k: float(v) 
            for k, v in trainer.callback_metrics.items() 
            if isinstance(v, (int, float))
        }
        
        # Always track metrics
        self.history.append({"epoch": epoch, **metrics})

        # Update UI periodically
        if epoch % self.update_every_n_epochs == 0:
            self._update_display()

    def _update_display(self) -> None:
        """Update the Marimo display."""
        if not self.history:
            return

        df = pl.DataFrame(self.history)
        
        # Melt for Altair
        metric_cols = [c for c in df.columns if c != "epoch"]
        if not metric_cols:
            return
            
        df_long = df.unpivot(index=["epoch"], on=metric_cols, variable_name="metric", value_name="value")
        
        chart = (
            alt.Chart(df_long.to_pandas())
            .mark_line()
            .encode(
                x="epoch:Q",
                y="value:Q",
                color="metric:N",
                tooltip=["epoch", "metric", "value"],
            )
            .properties(title="Training Metrics")
            .interactive()
        )
        
        self.chart_container.value = chart

        # Optional: Show latest metrics as a small table/stat
        latest = self.history[-1]
        stats = [mo.stat(k, f"{v:.4f}") for k, v in latest.items() if k != "epoch"]
        self.table_container.value = mo.hstack(stats, wrap=True)

