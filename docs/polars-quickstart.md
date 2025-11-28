# Polars Reference - Quickstart & Core Concepts

**Last Updated**: 2025-11-21
**Source Version**: Latest (pola-rs/polars)
**Status**: Current

## What is Polars?

Polars is a blazingly fast DataFrame library written in Rust and available for Python, JavaScript, R, and other languages. It's designed as a modern, high-performance alternative to pandas with:

- **Performance**: 10-100x faster than pandas through Rust implementation and Apache Arrow memory model
- **Lazy Evaluation**: Query optimization before execution
- **Eager Evaluation**: Traditional immediate execution when needed
- **Memory Efficient**: Columnar format with efficient compression
- **Expressive API**: Powerful expression language for complex transformations

## Quick Reference

### Installation

```bash
# Basic installation
pip install polars

# With optional dependencies (numpy, fsspec, etc.)
pip install 'polars[numpy,fsspec]'

# Using uv
uv add polars

# Using conda
conda install -c conda-forge polars
```

### Basic DataFrame Operations

```python
import polars as pl

# Create DataFrame
df = pl.DataFrame({
    "integers": [1, 2, 3, 4, 5],
    "floats": [1.0, 2.1, 3.2, 4.3, 5.4],
    "strings": ["a", "b", "c", "d", "e"]
})

# Read from CSV
df = pl.read_csv("data.csv")

# Read lazily (deferred execution)
lf = pl.scan_csv("data.csv")

# Basic display
print(df)
print(df.head(3))
print(df.schema)  # Column types
```

## Core Concepts

### 1. Eager vs Lazy Evaluation

**Eager API** - Executes immediately:

```python
import polars as pl

# Read and process immediately
df = pl.read_csv("data.csv")
filtered = df.filter(pl.col("value") > 100)
grouped = filtered.group_by("category").agg(pl.sum("amount"))
result = grouped.collect()  # Already executed
```

**Lazy API** - Defers execution for optimization:

```python
import polars as pl

# Build query plan without executing
lf = (
    pl.scan_csv("data.csv")
    .filter(pl.col("value") > 100)
    .group_by("category")
    .agg(pl.sum("amount"))
)

# Execute with optimization
result = lf.collect()  # Polars optimizes before execution
```

**Key Differences:**
- Lazy API allows Polars to optimize the entire query plan
- Eager API is useful for interactive exploration
- Mix both: `df.lazy()` converts DataFrame to LazyFrame, `lf.collect()` converts back

### 2. Expressions - The Core Language

Polars expressions are the foundation for all transformations:

```python
import polars as pl

# Basic expression
pl.col("column_name")

# With operations
pl.col("price") * 1.1  # 10% markup
pl.col("date").dt.year()  # Extract year from date
pl.col("name").str.to_uppercase()  # String operations

# Aggregations
pl.col("amount").sum()
pl.col("value").mean()
pl.col("id").n_unique()

# Conditional
pl.when(pl.col("status") == "active").then(1).otherwise(0)

# Multiple columns
pl.all()  # All columns
pl.exclude("id")  # All except id
pl.columns(["col1", "col2"])  # Specific columns
```

### 3. Select - Choose and Transform Columns

```python
import polars as pl

df = pl.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6],
    "c": ["x", "y", "z"]
})

# Select specific columns
df.select("a")
df.select(["a", "b"])
df.select(pl.col(["a", "b"]))

# Transform while selecting
df.select(
    pl.col("a"),
    (pl.col("b") * 2).alias("b_doubled"),
    pl.col("c").str.to_uppercase().alias("c_upper")
)

# Using expressions
df.select(pl.all().exclude("c"))  # All numeric columns
```

Output:
```
shape: (3, 3)
┌─────┬───────────┬─────────┐
│ a   ┆ b_doubled ┆ c_upper │
│ --- ┆ ---       ┆ ---     │
│ i64 ┆ i64       ┆ str     │
╞═════╪═══════════╪═════════╡
│ 1   ┆ 8         ┆ X       │
│ 2   ┆ 10        ┆ Y       │
│ 3   ┆ 12        ┆ Z       │
└─────┴───────────┴─────────┘
```

### 4. Filter - Select Rows

```python
import polars as pl

df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "salary": [50000, 60000, 75000]
})

# Simple filter
df.filter(pl.col("age") > 27)

# Multiple conditions (AND)
df.filter(
    (pl.col("age") > 25) & (pl.col("salary") > 55000)
)

# OR conditions
df.filter(
    (pl.col("name") == "Alice") | (pl.col("name") == "Bob")
)

# NOT
df.filter(~(pl.col("age") < 30))

# Using is_in
df.filter(pl.col("name").is_in(["Alice", "Charlie"]))

# String operations
df.filter(pl.col("name").str.starts_with("A"))
```

### 5. With Columns - Add or Update Columns

```python
import polars as pl

df = pl.DataFrame({
    "a": [1, 2, 3],
    "b": [4, 5, 6]
})

# Add new column
df.with_columns(
    (pl.col("a") + pl.col("b")).alias("sum_ab")
)

# Update existing column
df.with_columns(
    pl.col("a").mul(2).alias("a")
)

# Multiple new columns
df.with_columns(
    (pl.col("a") * 2).alias("a_doubled"),
    (pl.col("b") * 3).alias("b_tripled"),
    (pl.col("a") + pl.col("b")).alias("total")
)

# Conditional columns
df.with_columns(
    pl.when(pl.col("a") > 1)
    .then("high")
    .otherwise("low")
    .alias("category")
)
```

### 6. Group By & Aggregation

```python
import polars as pl

df = pl.DataFrame({
    "category": ["A", "B", "A", "B", "A"],
    "value": [10, 20, 30, 40, 50],
    "amount": [100, 200, 150, 300, 250]
})

# Group and aggregate
df.group_by("category").agg(
    pl.col("value").sum().alias("total_value"),
    pl.col("amount").mean().alias("avg_amount"),
    pl.col("*").count().alias("count")
)

# Multiple group columns
df.group_by(["category", "another_col"]).agg(
    pl.col("value").sum()
)

# Multiple aggregations per column
df.group_by("category").agg(
    pl.col("value").sum(),
    pl.col("value").mean(),
    pl.col("value").max()
)

# Available aggregations:
# .sum(), .mean(), .median(), .std(), .var()
# .min(), .max(), .first(), .last()
# .count(), .n_unique(), .arg_max(), .arg_min()
```

### 7. Join Operations

```python
import polars as pl

df1 = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"]
})

df2 = pl.DataFrame({
    "id": [1, 2, 4],
    "salary": [50000, 60000, 70000]
})

# Inner join (only matching)
df1.join(df2, on="id", how="inner")

# Left join (keep all from left)
df1.join(df2, on="id", how="left")

# Outer join (all from both)
df1.join(df2, on="id", how="outer")

# Join on different column names
df1.join(df2, left_on="id", right_on="emp_id")

# Multiple join keys
df1.join(df2, on=["id", "name"])
```

### 8. Type Casting

```python
import polars as pl

df = pl.DataFrame({
    "integers": [1, 2, 3],
    "floats": [1.0, 2.1, 3.2],
})

# Cast to different types
df.with_columns(
    pl.col("integers").cast(pl.Float32),
    pl.col("floats").cast(pl.Int64)
)

# Available types:
# Numeric: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64
#          Float32, Float64
# String: Utf8, Categorical
# Temporal: Date, Time, Duration, Datetime
# Other: Boolean, Null, List, Struct
```

### 9. String Operations

```python
import polars as pl

df = pl.DataFrame({
    "text": ["hello world", "POLARS", "Data Frame"]
})

df.select(
    pl.col("text").str.to_uppercase(),      # HELLO WORLD
    pl.col("text").str.to_lowercase(),      # hello world
    pl.col("text").str.lengths(),            # Length of each string
    pl.col("text").str.contains("world"),   # Check if contains
    pl.col("text").str.starts_with("hello"), # Starts with
    pl.col("text").str.extract(r"(\w+)", 0), # Regex extraction
    pl.col("text").str.replace("o", "0"),   # Replace
)
```

### 10. Date/Time Operations

```python
import polars as pl

df = pl.DataFrame({
    "date": ["2024-01-15", "2024-06-20", "2024-12-25"]
}).with_columns(
    pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
)

df.select(
    pl.col("date"),
    pl.col("date").dt.year().alias("year"),      # 2024
    pl.col("date").dt.month().alias("month"),    # 1, 6, 12
    pl.col("date").dt.day().alias("day"),        # 15, 20, 25
    pl.col("date").dt.strftime("%Y-%m").alias("year_month"),  # Format
)
```

## Common Patterns

### Pattern: Data Cleaning Pipeline

```python
import polars as pl

# Define pipeline
pipeline = (
    pl.read_csv("raw_data.csv")
    .filter(pl.col("age") > 0)  # Remove invalid age
    .with_columns(
        pl.col("salary").fill_nan(0),  # Handle NaN
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    .filter(pl.col("email").is_not_null())  # Remove null emails
    .drop_duplicates(subset=["id"])  # Remove duplicates
    .sort("date")
)

result = pipeline.collect()
```

### Pattern: Aggregation with Multiple Functions

```python
import polars as pl

df = pl.DataFrame({
    "product": ["A", "B", "A", "B"],
    "revenue": [100, 150, 200, 180],
    "quantity": [5, 3, 8, 6]
})

summary = df.group_by("product").agg(
    pl.col("revenue").sum().alias("total_revenue"),
    pl.col("revenue").mean().alias("avg_revenue"),
    pl.col("quantity").sum().alias("total_qty"),
    pl.col("*").count().alias("transactions")
).sort("total_revenue", descending=True)
```

### Pattern: Conditional Column Creation

```python
import polars as pl

df = pl.DataFrame({
    "score": [45, 55, 65, 75, 85, 95]
})

df.with_columns(
    pl.when(pl.col("score") >= 90)
    .then("A")
    .when(pl.col("score") >= 80)
    .then("B")
    .when(pl.col("score") >= 70)
    .then("C")
    .when(pl.col("score") >= 60)
    .then("D")
    .otherwise("F")
    .alias("grade")
)
```

### Pattern: Window Functions

```python
import polars as pl

df = pl.DataFrame({
    "group": ["A", "A", "B", "B"],
    "value": [10, 20, 30, 40]
})

df.with_columns(
    # Running sum per group
    pl.col("value").cum_sum().over("group").alias("running_sum"),
    # Rank within group
    pl.col("value").rank().over("group").alias("rank"),
    # Group mean
    pl.col("value").mean().over("group").alias("group_mean")
)
```

## Best Practices

### ✅ DO: Use Lazy API for Large Datasets

```python
# GOOD - Lazy evaluation allows optimization
result = (
    pl.scan_csv("huge_file.csv")
    .filter(pl.col("value") > 100)
    .group_by("category")
    .agg(pl.col("amount").sum())
    .collect()
)
```

### ❌ DON'T: Load Entire File Then Filter

```python
# BAD - Loads everything into memory first
df = pl.read_csv("huge_file.csv")
result = df.filter(pl.col("value") > 100)
```

### ✅ DO: Use .alias() for Clarity

```python
# GOOD - Clear column names
df.with_columns(
    (pl.col("price") * 0.9).alias("discounted_price")
)
```

### ✅ DO: Chain Operations for Readability

```python
# GOOD - Clear pipeline
result = (
    df
    .filter(pl.col("active") == True)
    .group_by("category")
    .agg(pl.col("revenue").sum())
    .sort("revenue", descending=True)
)
```

### ✅ DO: Use .explain() to Debug Queries

```python
# GOOD - See query optimization
query = pl.scan_csv("data.csv").filter(pl.col("value") > 100)
print(query.explain(optimized=True))
query.collect()
```

### ✅ DO: Leverage Built-in Data Types

```python
# GOOD - Use Polars types for efficiency
df = pl.DataFrame({
    "date": ["2024-01-15"],
    "amount": [1000],
}).with_columns(
    pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
    pl.col("amount").cast(pl.Int64)
)
```

### ❌ DON'T: Use .apply() for Complex Logic

```python
# BAD - Slow, Python-based processing
df.with_columns(
    pl.col("value").apply(lambda x: expensive_function(x))
)

# GOOD - Use native Polars expressions
df.with_columns(
    pl.when(pl.col("value") > 100).then(...).otherwise(...)
)
```

## Common Issues & Solutions

### Issue: "Column not found" Error

**Problem**: Reference to column that doesn't exist.

**Solution**: Check column names are correct (case-sensitive).

```python
df.columns  # List all column names
df.select(pl.col("exact_name"))
```

### Issue: Type Mismatch in Operations

**Problem**: `InvalidOperationError` when combining incompatible types.

**Solution**: Cast columns to compatible types first.

```python
df.with_columns(
    pl.col("string_col").cast(pl.Int64)
)
```

### Issue: Null Values Breaking Aggregations

**Problem**: `NULL` values in aggregations.

**Solution**: Use `.fill_null()` or `.drop_nulls()`.

```python
# Fill nulls with default
df.with_columns(pl.col("value").fill_null(0))

# Drop rows with nulls
df.drop_nulls(subset=["value"])
```

### Issue: Performance Issues with Large Files

**Problem**: Memory usage or slow execution.

**Solution**: Use lazy API for query optimization.

```python
# Lazy query gets optimized before collection
result = pl.scan_csv("file.csv").filter(...).collect()
```

## API Reference - Quick Lookup

### Reading Data

```python
pl.read_csv("file.csv")
pl.read_parquet("file.parquet")
pl.read_excel("file.xlsx")
pl.scan_csv("file.csv")  # Lazy
pl.scan_parquet("file.parquet")  # Lazy
```

### Column Selection & Transformation

```python
pl.col("name")              # Single column
pl.col(["a", "b"])          # Multiple columns
pl.all()                    # All columns
pl.exclude("id")            # All except
pl.columns(regex="^x_")     # Regex match
```

### Filtering

```python
df.filter(pl.col("age") > 25)
df.filter(pl.col("name") == "Alice")
df.filter(pl.col("value").is_null())
df.filter(pl.col("name").is_in(["A", "B"]))
```

### Aggregation Functions

```python
pl.col("x").sum()
pl.col("x").mean()
pl.col("x").min() / .max()
pl.col("x").first() / .last()
pl.col("x").count() / .n_unique()
```

### String Operations

```python
.str.to_uppercase()
.str.to_lowercase()
.str.lengths()
.str.contains(pattern)
.str.starts_with(prefix)
.str.extract(pattern, group)
.str.replace(pattern, value)
```

### Date/Time Operations

```python
.dt.year() / .month() / .day()
.dt.hour() / .minute() / .second()
.dt.strftime(format)
.dt.truncate(interval)
```

## Additional Resources

- **Official Docs**: https://docs.pola.rs/
- **Python API Reference**: https://docs.pola.rs/api/python/stable/
- **GitHub**: https://github.com/pola-rs/polars
- **SQL Support**: Polars supports SQL queries with `pl.sql()`
- **Performance Benchmarks**: https://www.pola.rs/

## Performance Tips

1. **Use lazy evaluation** for large datasets
2. **Select only needed columns** early in pipeline
3. **Filter early** to reduce data size
4. **Use built-in aggregations** instead of custom Python
5. **Leverage data types** - use appropriate types for memory efficiency
6. **Use partitioned Parquet** for massive datasets
7. **Enable streaming** for extremely large files: `.scan_csv(..., streaming=True)`
