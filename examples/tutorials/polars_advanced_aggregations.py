"""
Polars Advanced Aggregations - Complex Group By and Pivots

Based on docs/polars-quickstart.md Patterns 2-3
Demonstrates:
- Multiple aggregation functions
- Conditional aggregations
- Pivot tables
- Cross-tabulations
- Group by with expressions
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
        # Polars Advanced Aggregations

        **Complex group by** operations with multiple aggregations.

        ## Covered
        - 📊 Multiple agg functions
        - 🎯 Conditional aggregations
        - 📋 Pivot tables
        - 🔄 Cross-tabulations
        """
    )
    return


@app.cell
def _():
    """Import libraries"""
    import polars as pl
    import numpy as np
    from datetime import datetime, timedelta

    return datetime, np, pl, timedelta


@app.cell
def _(datetime, np, pl, timedelta):
    """Generate sales data"""
    np.random.seed(42)
    n_records = 500

    dates = [datetime(2024, 1, 1) + timedelta(days=i % 90) for i in range(n_records)]
    products = np.random.choice(["Laptop", "Mouse", "Keyboard", "Monitor"], n_records)
    regions = np.random.choice(["North", "South", "East", "West"], n_records)
    quantities = np.random.randint(1, 20, n_records)
    prices = {
        "Laptop": 1000,
        "Mouse": 25,
        "Keyboard": 75,
        "Monitor": 300,
    }

    df = pl.DataFrame(
        {
            "date": dates,
            "product": products,
            "region": regions,
            "quantity": quantities,
            "price": [prices[p] for p in products],
        }
    ).with_columns((pl.col("quantity") * pl.col("price")).alias("revenue"))

    print(f"✅ Generated {len(df)} sales records")
    return (df,)


@app.cell
def _(df, mo):
    """Preview data"""
    mo.md(f"## Sample Data\n\n{mo.ui.table(df.head(10))}")
    return


@app.cell
def _(mo):
    """Aggregation type selector"""
    agg_type = mo.ui.dropdown(
        options=["Multiple Functions", "Conditional Agg", "Pivot Table", "Complex Group By"],
        value="Multiple Functions",
        label="📊 Aggregation Type",
    )
    mo.md(f"## Aggregation\n\n{agg_type}")
    return (agg_type,)


@app.cell
def _(agg_type, df, pl):
    """Perform aggregation"""

    if agg_type.value == "Multiple Functions":
        # Group by with multiple aggregations
        result = df.group_by("product").agg(
            [
                pl.col("quantity").sum().alias("total_quantity"),
                pl.col("quantity").mean().round(2).alias("avg_quantity"),
                pl.col("revenue").sum().alias("total_revenue"),
                pl.col("revenue").max().alias("max_revenue"),
                pl.count().alias("num_transactions"),
            ]
        ).sort("total_revenue", descending=True)

        explanation = "Multiple aggregation functions on grouped data"

    elif agg_type.value == "Conditional Agg":
        # Conditional aggregations
        result = df.group_by("region").agg(
            [
                # High value transactions (revenue > 1000)
                pl.col("revenue").filter(pl.col("revenue") > 1000).sum().alias("high_value_revenue"),
                pl.col("revenue").filter(pl.col("revenue") > 1000).count().alias("high_value_count"),
                # Low value transactions
                pl.col("revenue").filter(pl.col("revenue") <= 100).sum().alias("low_value_revenue"),
                # Total
                pl.col("revenue").sum().alias("total_revenue"),
            ]
        )

        explanation = "Conditional aggregations using filter()"

    elif agg_type.value == "Pivot Table":
        # Pivot table: products vs regions
        result = (
            df.group_by(["product", "region"])
            .agg(pl.col("revenue").sum().alias("revenue"))
            .pivot(index="product", on="region", values="revenue")
            .fill_null(0)
        )

        explanation = "Pivot table showing revenue by product and region"

    else:  # Complex Group By
        # Multi-level grouping with expressions
        result = (
            df.with_columns(pl.col("date").dt.month().alias("month"))
            .group_by(["month", "product"])
            .agg(
                [
                    pl.col("revenue").sum().alias("monthly_revenue"),
                    pl.col("quantity").sum().alias("monthly_quantity"),
                    (pl.col("revenue").sum() / pl.col("quantity").sum())
                    .round(2)
                    .alias("avg_price"),
                ]
            )
            .sort("month", "monthly_revenue", descending=[False, True])
        )

        explanation = "Complex multi-level grouping with computed metrics"

    return (explanation, result)


@app.cell
def _(explanation, mo, result):
    """Display results"""
    mo.md(
        f"""
        ### Results

        **{explanation}**

        {mo.ui.table(result)}
        """
    )
    return


@app.cell
def _(mo):
    """Code patterns"""
    mo.md(
        r"""
        ## 💻 Aggregation Patterns

        ### Multiple Functions
        ```python
        df.group_by("category").agg([
            pl.col("value").sum(),
            pl.col("value").mean(),
            pl.count()
        ])
        ```

        ### Conditional
        ```python
        df.group_by("group").agg(
            pl.col("value").filter(pl.col("value") > 100).sum()
        )
        ```

        ### Pivot
        ```python
        df.pivot(index="row", on="col", values="value")
        ```

        See `docs/polars-quickstart.md` for more patterns.
        """
    )
    return


if __name__ == "__main__":
    app.run()
