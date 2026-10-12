"""Run the e-commerce sales analytics and display their results."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from src import config
from src.spark_analytics import SalesAnalytics

LOGGER = logging.getLogger(__name__)


def _display_timed_result(
    operation_name: str,
    operation: Callable[[], DataFrame],
) -> None:
    """Run an analytics operation, display its result, and report elapsed time."""
    start_time = time.perf_counter()
    result = operation()
    result.show()
    print(f"{operation_name} execution time: {time.perf_counter() - start_time:.2f} seconds")


def run_analytics(raw_data_dir: Path = config.DEFAULT_CONFIG.raw_data_dir) -> None:
    """Load the raw Parquet datasets, run each analysis, and stop Spark."""
    analytics = SalesAnalytics()
    session_start = time.perf_counter()
    spark: SparkSession = analytics.create_spark_session()
    print(f"Spark session creation time: {time.perf_counter() - session_start:.2f} seconds")

    try:
        datasets: dict[str, DataFrame] = {}
        for dataset_name in ("customers", "products", "orders"):
            parquet_path = raw_data_dir / f"{dataset_name}.parquet"
            load_start = time.perf_counter()
            datasets[dataset_name] = analytics.load_parquet(parquet_path)
            load_seconds = time.perf_counter() - load_start
            print(
                f"{dataset_name.title()} Parquet DataFrame setup time: "
                f"{load_seconds:.2f} seconds (Spark reads data lazily)"
            )

        orders = datasets["orders"]
        products = datasets["products"]

        _display_timed_result(
            "Top customers by revenue",
            lambda: analytics.top_customers_by_revenue(orders, products),
        )
        _display_timed_result(
            "Sales by category",
            lambda: analytics.sales_by_category(orders, products),
        )
        _display_timed_result(
            "Monthly trends",
            lambda: analytics.monthly_trends(orders, products),
        )
    finally:
        spark.stop()
        print("Spark session stopped.")


def main() -> int:
    """Run the analytics script and report unexpected failures."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run_analytics()
    except Exception:
        LOGGER.exception("Sales analytics execution failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
