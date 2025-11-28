# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pyzmq",
# ]
# ///
"""
Polars Window Functions - Advanced Time Series and Ranking Operations

Based on docs/polars-quickstart.md Pattern 4
Demonstrates:
- Ranking functions (rank, dense_rank, row_number)
- Rolling aggregations (rolling_mean, rolling_sum)
- Cumulative statistics (cumsum, cumcount)
- Partition-based window operations
- Shift operations (lag, lead)
"""

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Polars Window Functions

    **Window functions** perform calculations across rows related to the current row.
    Perfect for time series analysis, rankings, and moving averages.

    ## What You'll Learn
    - 📊 Ranking (rank, dense_rank, row_number)
    - 📈 Rolling aggregations (moving averages, sums)
    - 🔢 Cumulative statistics
    - 🎯 Partition-based operations
    - ⏰ Time-based windows
    """)
    return


@app.cell
def _():
    """Import required libraries"""
    import polars as pl
    import numpy as np
    from datetime import datetime, timedelta
    import altair as alt
    return alt, datetime, np, pl, timedelta


@app.cell
def _(datetime, np, pl, timedelta):
    """Generate sample sales data"""

    # Sales data for multiple products over time
    n_days = 180
    products = ["Product A", "Product B", "Product C"]
    dates = [datetime.now() - timedelta(days=i) for i in range(n_days)][::-1]

    data = []
    for product in products:
        base_sales = {"Product A": 100, "Product B": 150, "Product C": 80}[product]
        trend = {"Product A": 0.5, "Product B": 0.3, "Product C": 0.7}[product]

        for i, date in enumerate(dates):
            # Add trend, seasonality, and noise
            sales = (
                base_sales
                + trend * i
                + 20 * np.sin(i / 7 * np.pi)  # Weekly pattern
                + np.random.normal(0, 10)
            )
            sales = max(0, sales)  # No negative sales

            data.append(
                {
                    "date": date,
                    "product": product,
                    "sales": round(sales, 2),
                    "cost": round(sales * np.random.uniform(0.4, 0.6), 2),
                }
            )

    df_sales = pl.DataFrame(data).with_columns(
        pl.col("date").cast(pl.Date),
        profit=(pl.col("sales") - pl.col("cost")).round(2),
    )

    print(f"✅ Generated {len(df_sales)} sales records for {len(products)} products")
    return (df_sales,)


@app.cell
def _(df_sales, mo):
    """Data preview"""
    mo.md(
        f"""
        ## 📊 Sample Data

        Sales data for 3 products over 180 days:

        {mo.ui.table(df_sales.head(10))}

        **Total Records:** {len(df_sales):,} rows
        """
    )
    return


@app.cell
def _(mo):
    """Select window function operation"""

    operation = mo.ui.dropdown(
        options=[
            "Ranking Functions",
            "Rolling Aggregations",
            "Cumulative Statistics",
            "Partition Operations",
            "Lag/Lead Operations",
        ],
        value="Rolling Aggregations",
        label="📈 Choose Window Operation",
    )

    mo.md(f"## ⚙️ Window Function Selection\n\n{operation}")
    return (operation,)


@app.cell
def _(df_sales, operation, pl):
    """Apply selected window function"""

    if operation.value == "Ranking Functions":
        # Rank products by daily sales
        result = df_sales.with_columns(
            [
                # rank: Traditional ranking (1, 2, 2, 4)
                pl.col("sales")
                .rank(method="min", descending=True)
                .over("date")
                .alias("rank"),
                # dense_rank: No gaps in ranking (1, 2, 2, 3)
                pl.col("sales")
                .rank(method="dense", descending=True)
                .over("date")
                .alias("dense_rank"),
                # row_number: Sequential numbering (1, 2, 3, 4)
                pl.col("sales").rank(method="ordinal", descending=True).over("date").alias("row_number"),
            ]
        ).sort("date", "rank")

        explanation = """
        **Ranking Functions:**
        - `rank`: Ties get same rank, next rank skips (1, 2, 2, 4)
        - `dense_rank`: Ties get same rank, no gaps (1, 2, 2, 3)
        - `row_number`: Each row gets unique sequential number

        **over("date")**: Ranks are computed within each date
        """

    elif operation.value == "Rolling Aggregations":
        # Rolling 7-day and 30-day averages per product
        result = (
            df_sales.sort("product", "date")
            .with_columns(
                [
                    # 7-day rolling average
                    pl.col("sales")
                    .rolling_mean(window_size=7, min_periods=1)
                    .over("product")
                    .round(2)
                    .alias("sales_7d_avg"),
                    # 30-day rolling average
                    pl.col("sales")
                    .rolling_mean(window_size=30, min_periods=1)
                    .over("product")
                    .round(2)
                    .alias("sales_30d_avg"),
                    # 7-day rolling sum
                    pl.col("sales")
                    .rolling_sum(window_size=7, min_periods=1)
                    .over("product")
                    .round(2)
                    .alias("sales_7d_sum"),
                    # 7-day rolling std dev
                    pl.col("sales")
                    .rolling_std(window_size=7, min_periods=1)
                    .over("product")
                    .round(2)
                    .alias("sales_7d_std"),
                ]
            )
        )

        explanation = """
        **Rolling Aggregations:**
        - `rolling_mean(window_size=7)`: 7-day moving average
        - `rolling_sum()`: Cumulative sum over window
        - `rolling_std()`: Rolling standard deviation

        **over("product")**: Calculations done separately per product
        **min_periods=1**: Allow calculations with fewer than window_size points
        """

    elif operation.value == "Cumulative Statistics":
        # Cumulative sums and counts
        result = (
            df_sales.sort("product", "date")
            .with_columns(
                [
                    # Cumulative sum of sales per product
                    pl.col("sales").cum_sum().over("product").round(2).alias("cumulative_sales"),
                    # Cumulative profit
                    pl.col("profit").cum_sum().over("product").round(2).alias("cumulative_profit"),
                    # Running count
                    pl.col("sales").cum_count().over("product").alias("day_number"),
                    # Running average
                    (pl.col("sales").cum_sum() / pl.col("sales").cum_count())
                    .over("product")
                    .round(2)
                    .alias("running_average"),
                ]
            )
        )

        explanation = """
        **Cumulative Statistics:**
        - `cum_sum()`: Running total (accumulated sum)
        - `cum_count()`: Running count of rows
        - `running_average`: Cumulative sum ÷ cumulative count

        **Use Cases:**
        - Track total sales over time
        - Calculate year-to-date metrics
        - Monitor running averages
        """

    elif operation.value == "Partition Operations":
        # Statistics within partitions
        result = df_sales.with_columns(
            [
                # Daily total sales (all products)
                pl.col("sales").sum().over("date").round(2).alias("daily_total"),
                # Product's share of daily sales
                (pl.col("sales") / pl.col("sales").sum().over("date") * 100)
                .round(2)
                .alias("daily_share_pct"),
                # Product's rank for the day
                pl.col("sales").rank(descending=True).over("date").alias("daily_rank"),
                # Deviation from product's mean
                (pl.col("sales") - pl.col("sales").mean().over("product"))
                .round(2)
                .alias("deviation_from_mean"),
            ]
        ).sort("date", "daily_rank")

        explanation = """
        **Partition Operations:**
        - `over("date")`: Compute statistics per date
        - `over("product")`: Compute statistics per product
        - Combine multiple partitions for complex analysis

        **Example:**
        - Daily share = product sales ÷ total daily sales
        - Shows each product's contribution to daily revenue
        """

    else:  # Lag/Lead Operations
        # Previous and next values
        result = (
            df_sales.sort("product", "date")
            .with_columns(
                [
                    # Previous day's sales
                    pl.col("sales").shift(1).over("product").alias("prev_day_sales"),
                    # Next day's sales
                    pl.col("sales").shift(-1).over("product").alias("next_day_sales"),
                    # Day-over-day change
                    (pl.col("sales") - pl.col("sales").shift(1).over("product"))
                    .round(2)
                    .alias("sales_change"),
                    # Day-over-day change percentage
                    (
                        (pl.col("sales") - pl.col("sales").shift(1).over("product"))
                        / pl.col("sales").shift(1).over("product")
                        * 100
                    )
                    .round(2)
                    .alias("sales_change_pct"),
                    # 7-day lagged sales
                    pl.col("sales").shift(7).over("product").alias("sales_7d_ago"),
                ]
            )
        )

        explanation = """
        **Lag/Lead Operations:**
        - `shift(1)`: Previous row (lag)
        - `shift(-1)`: Next row (lead)
        - `shift(n)`: N rows back

        **Use Cases:**
        - Calculate day-over-day changes
        - Compare to same day last week
        - Detect trends and anomalies
        """
    return explanation, result


@app.cell
def _(explanation, mo, result):
    """Display results"""
    mo.md(
        f"""
        ## 📊 Results

        {explanation}

        ### Data Preview
        {mo.ui.table(result.head(20))}

        **Shape:** {result.shape[0]:,} rows × {result.shape[1]} columns
        """
    )
    return


@app.cell
def _(alt, operation, pl, result):
    """Visualize window function results"""

    if operation.value == "Rolling Aggregations":
        # Plot sales with rolling averages
        chart = (
            alt.Chart(result.to_pandas())
            .transform_fold(
                ["sales", "sales_7d_avg", "sales_30d_avg"],
                as_=["metric", "value"],
            )
            .mark_line()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("value:Q", title="Sales"),
                color=alt.Color("metric:N", title="Metric"),
                strokeDash=alt.StrokeDash("metric:N"),
                facet=alt.Facet("product:N", columns=1),
            )
            .properties(width=700, height=150, title="Rolling Averages by Product")
        )

    elif operation.value == "Cumulative Statistics":
        # Plot cumulative sales
        chart = (
            alt.Chart(result.to_pandas())
            .mark_line(point=False)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("cumulative_sales:Q", title="Cumulative Sales"),
                color=alt.Color("product:N", title="Product"),
                tooltip=["date:T", "product:N", "cumulative_sales:Q"],
            )
            .properties(width=700, height=400, title="Cumulative Sales Over Time")
            .interactive()
        )

    elif operation.value == "Partition Operations":
        # Plot daily share percentage
        chart = (
            alt.Chart(result.to_pandas())
            .mark_area(opacity=0.7)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("daily_share_pct:Q", stack="normalize", title="Share of Daily Sales (%)"),
                color=alt.Color("product:N", title="Product"),
                tooltip=["date:T", "product:N", "daily_share_pct:Q"],
            )
            .properties(width=700, height=400, title="Product Share of Daily Sales")
        )

    elif operation.value == "Lag/Lead Operations":
        # Plot sales changes
        chart = (
            alt.Chart(result.filter(result["sales_change"].is_not_null()).to_pandas())
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("sales_change:Q", title="Day-over-Day Change"),
                color=alt.condition(
                    alt.datum.sales_change > 0,
                    alt.value("green"),
                    alt.value("red"),
                ),
                facet=alt.Facet("product:N", columns=1),
                tooltip=["date:T", "sales_change:Q", "sales_change_pct:Q"],
            )
            .properties(width=700, height=150, title="Day-over-Day Sales Changes")
        )

    else:  # Ranking
        # Show top-ranked products per day (sample)
        sample = result.filter(pl.col("date").is_in(result["date"].unique()[:30]))
        chart = (
            alt.Chart(sample.to_pandas())
            .mark_rect()
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("product:N", title="Product"),
                color=alt.Color("rank:O", scale=alt.Scale(scheme="blues"), title="Rank"),
                tooltip=["date:T", "product:N", "sales:Q", "rank:O"],
            )
            .properties(width=700, height=200, title="Daily Product Rankings (Sample)")
        )

    chart_output = chart
    return (chart_output,)


@app.cell
def _(chart_output, mo):
    """Display visualization"""
    mo.md(
        f"""
        ## 📈 Visualization

        {mo.ui.altair_chart(chart_output)}
        """
    )
    return


@app.cell
def _(mo):
    """Common window function patterns"""
    mo.md(
        f"""
        ## 🎯 Common Patterns Cheat Sheet

        ### Ranking
        ```python
        # Rank by sales (high to low)
        df.with_columns(
            pl.col("sales").rank(descending=True).over("date").alias("rank")
        )
        ```

        ### Rolling Average
        ```python
        # 7-day moving average per group
        df.sort("date").with_columns(
            pl.col("sales")
            .rolling_mean(window_size=7)
            .over("product")
            .alias("rolling_avg")
        )
        ```

        ### Cumulative Sum
        ```python
        # Running total per group
        df.sort("date").with_columns(
            pl.col("sales").cum_sum().over("product").alias("cumulative")
        )
        ```

        ### Lag/Lead
        ```python
        # Previous and next values
        df.sort("date").with_columns([
            pl.col("sales").shift(1).over("product").alias("previous"),
            pl.col("sales").shift(-1).over("product").alias("next")
        ])
        ```

        ### Percent of Total
        ```python
        # Share within partition
        df.with_columns(
            (pl.col("sales") / pl.col("sales").sum().over("date") * 100)
            .alias("share_pct")
        )
        ```

        ### Z-Score (Standardization)
        ```python
        # Standardize within group
        df.with_columns(
            ((pl.col("sales") - pl.col("sales").mean().over("product"))
             / pl.col("sales").std().over("product"))
            .alias("z_score")
        )
        ```

        ## 💡 Pro Tips

        1. **Always sort first** when using rolling/cumulative functions
        2. **Use `over()` for partitions** - calculations within groups
        3. **`min_periods` in rolling functions** - handles edge cases
        4. **Combine multiple windows** - complex multi-step analysis
        5. **Filter nulls** when using lag/lead (first/last rows)

        ## 📚 Learn More

        - [Polars Window Functions Docs](https://pola-rs.github.io/polars/py-polars/html/reference/expressions/window.html)
        - See `docs/polars-quickstart.md` for more patterns
        """
    )
    return


if __name__ == "__main__":
    app.run()
