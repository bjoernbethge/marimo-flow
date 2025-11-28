import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# DuckDB SQL Integration""")
    return


@app.cell
def _():
    """Import required libraries"""
    from datetime import datetime, timedelta

    import duckdb
    import numpy as np
    import polars as pl

    return datetime, duckdb, np, pl, timedelta


@app.cell
def _(mo):
    """Database configuration"""
    db_type = mo.ui.dropdown(
        options=["In-Memory", "File", "MotherDuck"],
        value="In-Memory",
        label="🗄️ Database Type"
    )

    db_path = mo.ui.text(
        label="📁 Database Path",
        value=":memory:",
        placeholder="path/to/database.db"
    )

    return db_path, db_type


@app.cell
def _(db_type):
    db_type.value
    return


@app.cell
def _():
    return


@app.cell
def _(conn, mo, products, temp):
    _df = mo.sql(
        f"""
        SELECT category FROM temp.products LIMIT 100
        """,
        engine=conn
    )
    return


@app.cell
def _(conn, mo, products, temp):
    _df = mo.sql(
        f"""
        SELECT * FROM temp.products LIMIT 100
        """,
        engine=conn
    )
    return


@app.cell
def _(db_path, db_type, mo):
    """Display configuration"""
    mo.md(f"""
    ## ⚙️ Database Configuration

    {db_type} {db_path}
    """)
    return


@app.cell
def _(db_path, db_type, duckdb):
    """Create DuckDB connection"""
    if db_type.value == "In-Memory":
        conn = duckdb.connect(":memory:")
    elif db_type.value == "File":
        conn = duckdb.connect(db_path.value)
    else:  # MotherDuck
        # This will prompt for authentication
        conn = duckdb.connect()
        conn.execute(f"ATTACH '{db_path.value}'")

    print(f"✅ Connected to {db_type.value} database")

    return (conn,)


@app.cell
def _(mo):
    """Sample data generation"""
    mo.md("""
    ## 📊 Create Sample Data

    Let's create some sample datasets to work with:
    """)
    return


@app.cell
def _(datetime, np, pl, timedelta):
    """Generate sample sales data"""
    # Create date range
    dates = pl.date_range(
        start=datetime.now() - timedelta(days=365),
        end=datetime.now(),
        freq='D'
    )

    # Generate sales data
    n_records = len(dates) * 10  # 10 sales per day average

    sales_df = pl.DataFrame({
        'order_id': pl.arange(1, n_records + 1),
        'order_date': np.random.choice(dates, n_records),
        'customer_id': np.random.randint(1, 1000, n_records),
        'product_id': np.random.randint(1, 100, n_records),
        'quantity': np.random.randint(1, 10, n_records),
        'unit_price': np.round(np.random.uniform(10, 1000, n_records), 2),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_records)
    })

    # Calculate total
    sales_df['total_amount'] = sales_df['quantity'] * sales_df['unit_price']

    print(f"✅ Generated {len(sales_df):,} sales records")

    return (sales_df,)


@app.cell
def _(np, pl):
    """Generate customer data"""
    customers_df = pl.DataFrame({
        'customer_id': pl.arange(1, 1000),
        'customer_name': [f'Customer {i}' for i in range(1, 1000)],
        'email': [f'customer{i}@example.com' for i in range(1, 1000)],
        'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Japan'], 999),
        'signup_date': pl.date_range(end='2024-01-01', periods=999),
        'is_premium': np.random.choice([True, False], 999, p=[0.2, 0.8])
    })

    print(f"✅ Generated {len(customers_df):,} customer records")

    return (customers_df,)


@app.cell
def _(np, pl):
    """Generate product data"""
    categories = ['Electronics', 'Clothing', 'Food', 'Books', 'Home']

    products_df = pl.DataFrame({
        'product_id': pl.arange(1, 100),
        'product_name': [f'Product {i}' for i in range(1, 100)],
        'category': np.random.choice(categories, 99),
        'cost': np.round(np.random.uniform(5, 500, 99), 2),
        'price': np.round(np.random.uniform(10, 1000, 99), 2),
        'stock_quantity': np.random.randint(0, 1000, 99)
    })

    print(f"✅ Generated {len(products_df):,} product records")

    return categories, products_df


@app.cell
def _(conn, customers_df, products_df, sales_df):
    """Register dataframes with DuckDB"""
    # Register pandas dataframes
    conn.register('sales', sales_df)
    conn.register('customers', customers_df)
    conn.register('products', products_df)

    print("✅ Dataframes registered with DuckDB")

    return


@app.cell
def _(mo):
    """SQL controls"""
    mo.md("""
    ## 🔍 Interactive SQL Queries

    Now let's run some interactive SQL queries with parameters:
    """)
    return


@app.cell
def _(categories, mo):
    """Query parameters"""
    date_range = mo.ui.date_range(
        label="📅 Date Range",
        start="2024-01-01",
        end="2024-12-31"
    )

    region_select = mo.ui.multiselect(
        options=['North', 'South', 'East', 'West'],
        value=['North', 'South'],
        label="🌍 Regions"
    )

    category_select = mo.ui.dropdown(
        options=['All'] + categories,
        value='All',
        label="📦 Category"
    )

    min_amount = mo.ui.slider(
        start=0, stop=10000, step=100, value=1000,
        label="💰 Minimum Order Amount"
    )

    return category_select, date_range, min_amount, region_select


@app.cell
def _(category_select, date_range, min_amount, mo, region_select):
    """Display query parameters"""
    mo.md(f"""
    ### 🎛️ Query Parameters

    {date_range} {region_select}
    {category_select} {min_amount}
    """)
    return


@app.cell
def _(category_select, date_range, min_amount, mo, region_select):
    """Sales summary query"""
    # Convert region list to SQL format
    regions_sql = "','".join(region_select.value)

    # Category filter
    category_filter = "" if category_select.value == "All" else f"AND p.category = '{category_select.value}'"

    sales_summary = mo.sql(
        f"""
        SELECT 
            DATE_TRUNC('month', s.order_date) as month,
            s.region,
            p.category,
            COUNT(DISTINCT s.order_id) as order_count,
            COUNT(DISTINCT s.customer_id) as unique_customers,
            SUM(s.quantity) as total_quantity,
            ROUND(SUM(s.total_amount), 2) as total_revenue,
            ROUND(AVG(s.total_amount), 2) as avg_order_value
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        WHERE s.order_date BETWEEN '{date_range.value[0]}' AND '{date_range.value[1]}'
            AND s.region IN ('{regions_sql}')
            AND s.total_amount >= {min_amount.value}
            {category_filter}
        GROUP BY month, s.region, p.category
        ORDER BY month DESC, total_revenue DESC
        """
    )

    return (sales_summary,)


@app.cell
def _(mo, sales_summary):
    """Display sales summary"""
    mo.md(f"""
    ### 📊 Sales Summary

    {mo.ui.table(sales_summary)}
    """)
    return


@app.cell
def _(mo):
    """Customer analysis"""
    mo.md("""
    ## 👥 Customer Analysis
    """)
    return


@app.cell
def _(mo):
    """Premium customer filter"""
    premium_only = mo.ui.checkbox(
        label="👑 Premium Customers Only",
        value=False
    )

    top_n = mo.ui.slider(
        start=5, stop=50, step=5, value=10,
        label="🏆 Top N Customers"
    )

    mo.md(f"{premium_only} {top_n}")

    return premium_only, top_n


@app.cell
def _(mo, premium_only, top_n):
    """Top customers query"""
    premium_filter = "AND c.is_premium = TRUE" if premium_only.value else ""

    top_customers = mo.sql(
        f"""
        WITH customer_stats AS (
            SELECT 
                c.customer_id,
                c.customer_name,
                c.country,
                c.is_premium,
                COUNT(DISTINCT s.order_id) as total_orders,
                ROUND(SUM(s.total_amount), 2) as lifetime_value,
                ROUND(AVG(s.total_amount), 2) as avg_order_value,
                MAX(s.order_date) as last_order_date,
                DATEDIFF('day', MAX(s.order_date), CURRENT_DATE) as days_since_last_order
            FROM customers c
            LEFT JOIN sales s ON c.customer_id = s.customer_id
            WHERE 1=1 {premium_filter}
            GROUP BY c.customer_id, c.customer_name, c.country, c.is_premium
        )
        SELECT *
        FROM customer_stats
        WHERE lifetime_value > 0
        ORDER BY lifetime_value DESC
        LIMIT {top_n.value}
        """
    )

    return (top_customers,)


@app.cell
def _(mo, top_customers):
    """Display top customers"""
    mo.md(f"""
    ### 🏆 Top Customers by Lifetime Value

    {mo.ui.table(top_customers)}
    """)
    return


@app.cell
def _(mo):
    """Product performance"""
    mo.md("""
    ## 📦 Product Performance Analysis
    """)
    return


@app.cell
def _(mo, products, sales):
    """Product analysis query"""
    product_performance = mo.sql(
        """
        WITH product_stats AS (
            SELECT 
                p.product_id,
                p.product_name,
                p.category,
                p.price,
                p.cost,
                p.stock_quantity,
                COUNT(DISTINCT s.order_id) as times_sold,
                SUM(s.quantity) as units_sold,
                ROUND(SUM(s.total_amount), 2) as total_revenue,
                ROUND(SUM(s.quantity * (p.price - p.cost)), 2) as total_profit,
                ROUND(AVG(s.quantity), 2) as avg_quantity_per_order
            FROM products p
            LEFT JOIN sales s ON p.product_id = s.product_id
            GROUP BY p.product_id, p.product_name, p.category, p.price, p.cost, p.stock_quantity
        )
        SELECT 
            *,
            ROUND(total_profit / NULLIF(total_revenue, 0) * 100, 2) as profit_margin_pct,
            CASE 
                WHEN stock_quantity = 0 THEN 'Out of Stock'
                WHEN stock_quantity < units_sold * 0.1 THEN 'Low Stock'
                ELSE 'In Stock'
            END as stock_status
        FROM product_stats
        ORDER BY total_revenue DESC
        """
    )

    return (product_performance,)


@app.cell
def _(mo, product_performance):
    """Display product performance"""
    mo.md(f"""
    ### 📊 Product Performance Metrics

    {mo.ui.table(product_performance)}
    """)
    return


@app.cell
def _(mo):
    """Time series analysis"""
    mo.md("""
    ## 📈 Time Series Analysis
    """)
    return


@app.cell
def _(mo):
    """Aggregation level"""
    time_grain = mo.ui.dropdown(
        options=['day', 'week', 'month', 'quarter'],
        value='month',
        label="📅 Time Granularity"
    )

    mo.md(f"{time_grain}")

    return (time_grain,)


@app.cell
def _(mo, sales, time_grain):
    """Time series query"""
    time_series = mo.sql(
        f"""
        SELECT 
            DATE_TRUNC('{time_grain.value}', order_date) as period,
            COUNT(DISTINCT order_id) as orders,
            COUNT(DISTINCT customer_id) as unique_customers,
            SUM(quantity) as units_sold,
            ROUND(SUM(total_amount), 2) as revenue,
            ROUND(AVG(total_amount), 2) as avg_order_value,
            ROUND(SUM(total_amount) / COUNT(DISTINCT customer_id), 2) as revenue_per_customer
        FROM sales
        GROUP BY period
        ORDER BY period
        """
    )

    return (time_series,)


@app.cell
def _(mo, time_series):
    """Display time series"""
    mo.md(f"""
    ### 📈 Sales Trends Over Time

    {mo.ui.table(time_series)}
    """)
    return


@app.cell
def _(mo):
    """Advanced SQL features"""
    mo.md("""
    ## 🚀 Advanced DuckDB Features

    ### Window Functions
    """)
    return


@app.cell
def _(mo, sales):
    """Window function example"""
    window_analysis = mo.sql(
        """
        WITH monthly_sales AS (
            SELECT 
                DATE_TRUNC('month', order_date) as month,
                region,
                SUM(total_amount) as monthly_revenue
            FROM sales
            GROUP BY month, region
        )
        SELECT 
            month,
            region,
            ROUND(monthly_revenue, 2) as revenue,
            ROUND(LAG(monthly_revenue, 1) OVER (PARTITION BY region ORDER BY month), 2) as prev_month_revenue,
            ROUND(monthly_revenue - LAG(monthly_revenue, 1) OVER (PARTITION BY region ORDER BY month), 2) as month_over_month_change,
            ROUND((monthly_revenue - LAG(monthly_revenue, 1) OVER (PARTITION BY region ORDER BY month)) / 
                  NULLIF(LAG(monthly_revenue, 1) OVER (PARTITION BY region ORDER BY month), 0) * 100, 2) as growth_rate_pct,
            RANK() OVER (PARTITION BY month ORDER BY monthly_revenue DESC) as region_rank
        FROM monthly_sales
        ORDER BY month DESC, revenue DESC
        """
    )

    return (window_analysis,)


@app.cell
def _(mo, window_analysis):
    """Display window analysis"""
    mo.md(f"""
    ### 📊 Month-over-Month Regional Performance

    {mo.ui.table(window_analysis)}
    """)
    return


@app.cell
def _(mo):
    """Export functionality"""
    mo.md("""
    ## 💾 Data Export

    DuckDB can export query results in various formats:
    """)
    return


@app.cell
def _(mo):
    """Export format selection"""
    export_format = mo.ui.dropdown(
        options=['CSV', 'Parquet', 'JSON', 'Excel'],
        value='CSV',
        label="📄 Export Format"
    )

    export_button = mo.ui.run_button(label="💾 Export Data")

    mo.md(f"{export_format} {export_button}")

    return export_button, export_format


@app.cell
def _(conn, export_button, export_format, mo, sales_summary):
    """Export data"""
    mo.stop(not export_button.value, "Click 'Export Data' to export")

    filename = f"sales_summary.{export_format.value.lower()}"

    try:
        if export_format.value == 'CSV':
            conn.execute(f"COPY sales_summary TO '{filename}' (HEADER, DELIMITER ',')")
        elif export_format.value == 'Parquet':
            conn.execute(f"COPY sales_summary TO '{filename}' (FORMAT PARQUET)")
        elif export_format.value == 'JSON':
            conn.execute(f"COPY sales_summary TO '{filename}' (FORMAT JSON)")
        else:  # Excel
            # For Excel, we need to use pandas
            sales_summary.to_excel(filename, index=False)

        mo.md(f"✅ Data exported to `{filename}`").callout(kind="success")
    except Exception as e:
        mo.md(f"❌ Export error: {e}").callout(kind="danger")

    return


@app.cell
def _(mo):
    """External data sources"""
    mo.md("""
    ## 🌐 External Data Sources

    DuckDB can query various external sources directly:

    ```sql
    -- Query CSV files
    SELECT * FROM read_csv('path/to/file.csv');

    -- Query Parquet files
    SELECT * FROM read_parquet('s3://bucket/file.parquet');

    -- Query JSON files
    SELECT * FROM read_json('https://api.example.com/data.json');

    -- Query from URLs
    SELECT * FROM 'https://raw.githubusercontent.com/data.csv';

    -- Query PostgreSQL
    ATTACH 'postgresql://user:pass@host:port/db' AS postgres_db;
    SELECT * FROM postgres_db.schema.table;
    ```
    """)
    return


@app.cell
def _(mo):
    """DuckDB extensions"""
    mo.md("""
    ## 🔌 DuckDB Extensions

    Load and use DuckDB extensions:
    """)
    return


@app.cell
def _(conn):
    """Install and load extensions"""
    # Common extensions
    extensions = ['httpfs', 'json', 'parquet']

    for ext in extensions:
        try:
            conn.execute(f"INSTALL {ext}")
            conn.execute(f"LOAD {ext}")
            print(f"✅ Loaded extension: {ext}")
        except Exception as e:
            print(f"ℹ️ Extension {ext}: {e}")

    return


@app.cell
def _(mo):
    """Tips and best practices"""
    mo.md("""
    ## 💡 Tips & Best Practices

    ### Performance
    - Use `EXPLAIN` to understand query plans
    - Create indexes on frequently queried columns
    - Use column selection instead of `SELECT *`
    - Leverage DuckDB's columnar storage

    ### SQL Cells
    - Reference Python variables with `{variable}`
    - Chain SQL cells by referencing output dataframes
    - Use `_df` prefix to make results private
    - Set output format: `native`, `polars`, or `pandas`

    ### Data Sources
    - Register DataFrames for complex joins
    - Use native DuckDB formats (Parquet) for best performance
    - Stream large files with `read_csv_auto`
    - Partition large datasets by date/category

    ### Integration
    - Combine SQL and Python for complex analysis
    - Use SQL for data transformation, Python for ML
    - Export results for sharing or further processing
    - Connect to cloud warehouses (MotherDuck, S3, etc.)
    """)
    return


if __name__ == "__main__":
    app.run()
