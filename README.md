# E-commerce Data Pipeline

A small, reproducible Python project that generates synthetic customers,
products, and orders, then uses PySpark to produce business insight datasets.
All generated records are fictional and intended for development and testing.

## Project structure

```text
.
├── data/
│   ├── processed/                 # Spark output datasets (generated)
│   └── raw/                       # Synthetic input Parquet files (generated)
├── notebooks/                     # Jupyter notebooks for exploration
├── main.py                        # Generate datasets and save Parquet files
├── src/
│   ├── __init__.py
│   ├── config.py                  # Paths and generation defaults
│   ├── data_generator.py          # Synthetic CSV data generator
│   └── spark_analytics.py         # PySpark transformations and outputs
├── tests/
│   ├── test_data_generator.py
│   └── test_main.py
├── .gitignore
└── requirements.txt
```

## Setup

Use Python 3.10 or newer and install a compatible Java runtime (Java 17 or
newer works with current PySpark releases). From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the pipeline

Generate the default dataset as Parquet files (100,000 customers, 10,000
products, and 1,000,000 orders):

```powershell
python main.py
```

The script reports generation time and file size for each dataset. The Parquet
files are written to `data/raw/customers.parquet`, `data/raw/products.parquet`,
and `data/raw/orders.parquet`. To customize counts or the seed, edit the
defaults in `src/config.py`.

The generator can also save CSV files when used directly from Python:

```python
from pathlib import Path
from src.data_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    customer_count=200,
    product_count=30,
    order_count=800,
    output_dir=Path("data/raw"),
    seed=7,
)
customers = generator.generate_customers()
products = generator.generate_products()
orders = generator.generate_orders()
generator.save_datasets(customers, products, orders)
```

Run the Spark analytics after generation:

```powershell
python run_analytics.py
```

The script loads the three Parquet files, displays top customers, sales by
category, and monthly revenue trends, and reports elapsed times. Spark reads
Parquet lazily, so the analytics operation timings include execution when
results are displayed.

Raw Parquet files are written to `data/raw/`. Spark writes three CSV insight
datasets as directories under `data/processed/`:

- `category_revenue/`: units sold and revenue by product category
- `monthly_sales/`: revenue and month-over-month revenue growth
- `top_customers/`: customers ranked by total revenue

Each Spark output directory contains a CSV part file and Spark's metadata files.
Generated data and analysis outputs are ignored by Git.

## Tests

Run the generator tests from the repository root:

```powershell
python -m pytest
```
