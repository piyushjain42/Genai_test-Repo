"""Generate synthetic e-commerce data and save it as Parquet files."""

import logging
import time
from pathlib import Path

from src import config
from src.data_generator import SyntheticDataGenerator

LOGGER = logging.getLogger(__name__)


def run_pipeline(settings: config.PipelineConfig = config.DEFAULT_CONFIG) -> dict[str, Path]:
    """Generate the configured datasets, save Parquet files, and report timings."""
    generator = SyntheticDataGenerator(
        customer_count=settings.customer_count,
        product_count=settings.product_count,
        order_count=settings.order_count,
        output_dir=settings.raw_data_dir,
        seed=settings.random_seed,
    )

    generation_start = time.perf_counter()
    customers = generator.generate_customers()
    customer_seconds = time.perf_counter() - generation_start

    generation_start = time.perf_counter()
    products = generator.generate_products()
    product_seconds = time.perf_counter() - generation_start

    generation_start = time.perf_counter()
    orders = generator.generate_orders()
    order_seconds = time.perf_counter() - generation_start

    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "customers": customers,
        "products": products,
        "orders": orders,
    }
    output_paths: dict[str, Path] = {}
    for name, dataframe in datasets.items():
        output_path = settings.raw_data_dir / f"{name}.parquet"
        dataframe.to_parquet(output_path, index=False, engine="pyarrow")
        output_paths[name] = output_path

    timings = {
        "customers": customer_seconds,
        "products": product_seconds,
        "orders": order_seconds,
    }
    total_seconds = sum(timings.values())
    print(f"Generation completed in {total_seconds:.2f} seconds.")
    for name, output_path in output_paths.items():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(
            f"{name.title()}: generated in {timings[name]:.2f} seconds; "
            f"saved to {output_path} ({file_size_mb:.2f} MB)"
        )
    return output_paths


def main() -> int:
    """Run the pipeline, reporting failures and returning a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run_pipeline()
    except Exception:
        LOGGER.exception("E-commerce data generation failed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
