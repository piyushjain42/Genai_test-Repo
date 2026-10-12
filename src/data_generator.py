"""Generate synthetic e-commerce customers, products, and orders."""

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from src.config import DEFAULT_CONFIG

LOGGER = logging.getLogger(__name__)

PRODUCT_CATEGORIES: Final[tuple[str, ...]] = (
    "Electronics",
    "Clothing",
    "Home",
    "Sports",
    "Books",
)
CUSTOMER_COLUMNS: Final[tuple[str, ...]] = (
    "customer_id",
    "name",
    "email",
    "age",
    "city",
    "country",
    "registration_date",
)
PRODUCT_COLUMNS: Final[tuple[str, ...]] = (
    "product_id",
    "name",
    "category",
    "price",
    "stock",
    "rating",
)
ORDER_COLUMNS: Final[tuple[str, ...]] = (
    "order_id",
    "customer_id",
    "product_id",
    "quantity",
    "order_date",
)


class SyntheticDataGenerator:
    """Generate reproducible synthetic e-commerce datasets as pandas DataFrames.

    Customer and product counts default to 100,000 and 10,000. The default
    order count is 1,000,000. Use smaller counts when experimenting locally.
    """

    def __init__(
        self,
        customer_count: int = DEFAULT_CONFIG.customer_count,
        product_count: int = DEFAULT_CONFIG.product_count,
        order_count: int = DEFAULT_CONFIG.order_count,
        output_dir: Path = DEFAULT_CONFIG.raw_data_dir,
        seed: int = DEFAULT_CONFIG.random_seed,
    ) -> None:
        """Initialize dataset sizes, output path, and reproducible random generators."""
        if customer_count <= 0 or product_count <= 0 or order_count <= 0:
            raise ValueError("Customer, product, and order counts must all be positive.")

        self.customer_count = customer_count
        self.product_count = product_count
        self.order_count = order_count
        self.output_dir = Path(output_dir)
        self.seed = seed
        self._faker = Faker()
        self._faker.seed_instance(seed)
        self._rng = np.random.default_rng(seed)
        self.customers: pd.DataFrame | None = None
        self.products: pd.DataFrame | None = None

    def generate_customers(self) -> pd.DataFrame:
        """Generate customers with Faker text, normally distributed ages, and dates."""
        LOGGER.info("Generating %d customers", self.customer_count)
        ages = np.clip(
            self._rng.normal(loc=35, scale=12, size=self.customer_count).round(),
            18,
            90,
        ).astype(np.int16)
        days_ago = self._rng.integers(0, 5 * 365 + 1, size=self.customer_count)
        registration_dates = (
            pd.Timestamp.today().normalize() - pd.to_timedelta(days_ago, unit="D")
        ).date

        customer_rows = (
            {
                "customer_id": customer_id,
                "name": self._faker.name(),
                "email": self._faker.unique.email(),
                "age": int(ages[customer_id - 1]),
                "city": self._faker.city(),
                "country": self._faker.country(),
                "registration_date": registration_dates[customer_id - 1],
            }
            for customer_id in tqdm(
                range(1, self.customer_count + 1),
                desc="Generating customers",
                unit="customer",
            )
        )
        self.customers = pd.DataFrame.from_records(customer_rows, columns=CUSTOMER_COLUMNS)
        LOGGER.info("Generated %d customers", len(self.customers))
        return self.customers

    def generate_products(self) -> pd.DataFrame:
        """Generate products with bounded prices, stock counts, and ratings."""
        LOGGER.info("Generating %d products", self.product_count)
        prices = np.clip(
            self._rng.lognormal(mean=4.1, sigma=0.85, size=self.product_count),
            10,
            500,
        ).round(2)
        stocks = np.clip(
            self._rng.negative_binomial(n=2, p=0.04, size=self.product_count),
            0,
            1_000,
        ).astype(np.int32)
        ratings = self._rng.choice(
            np.linspace(1, 5, num=10),
            size=self.product_count,
            p=np.array(
                [0.01, 0.01, 0.02, 0.02, 0.04, 0.05, 0.08, 0.12, 0.20, 0.45]
            ),
        )
        categories = self._rng.choice(PRODUCT_CATEGORIES, size=self.product_count)

        product_rows = (
            {
                "product_id": product_id,
                "name": f"{self._faker.word().title()} {self._faker.word().title()}",
                "category": categories[product_id - 1],
                "price": float(prices[product_id - 1]),
                "stock": int(stocks[product_id - 1]),
                "rating": float(ratings[product_id - 1]),
            }
            for product_id in tqdm(
                range(1, self.product_count + 1),
                desc="Generating products",
                unit="product",
            )
        )
        self.products = pd.DataFrame.from_records(product_rows, columns=PRODUCT_COLUMNS)
        LOGGER.info("Generated %d products", len(self.products))
        return self.products

    def generate_orders(self) -> pd.DataFrame:
        """Generate orders with valid foreign keys and Pareto-like customer demand.

        Customers are sampled with rank-based power-law weights. At the default
        customer count, the highest-ranked fifth receives about 80% of the
        sampling probability, while every order still references a valid customer.
        Generate customers and products before calling this method.
        """
        if self.customers is None or self.products is None:
            raise RuntimeError(
                "Generate customers and products before generating orders."
            )

        LOGGER.info("Generating %d orders", self.order_count)
        customer_ranks = np.arange(1, self.customer_count + 1, dtype=np.float64)
        customer_weights = np.power(customer_ranks, -0.92)
        customer_weights /= customer_weights.sum()
        customer_ids = self._rng.choice(
            self.customers["customer_id"].to_numpy(),
            size=self.order_count,
            p=customer_weights,
        )
        product_ids = self._rng.integers(
            1,
            self.product_count + 1,
            size=self.order_count,
            dtype=np.int32,
        )
        quantities = self._rng.integers(1, 11, size=self.order_count, dtype=np.int8)

        # Exponential day offsets make recent orders more common than old ones.
        days_ago = np.minimum(
            self._rng.exponential(scale=240, size=self.order_count).astype(np.int32),
            730,
        )
        order_dates = (
            pd.Timestamp.today().normalize() - pd.to_timedelta(days_ago, unit="D")
        ).date

        chunk_size = 100_000
        order_frames: list[pd.DataFrame] = []
        for start in tqdm(
            range(0, self.order_count, chunk_size),
            desc="Generating orders",
            unit="chunk",
        ):
            stop = min(start + chunk_size, self.order_count)
            order_frames.append(
                pd.DataFrame(
                    {
                        "order_id": np.arange(start + 1, stop + 1, dtype=np.int64),
                        "customer_id": customer_ids[start:stop],
                        "product_id": product_ids[start:stop],
                        "quantity": quantities[start:stop],
                        "order_date": order_dates[start:stop],
                    },
                    columns=ORDER_COLUMNS,
                )
            )

        orders = pd.concat(order_frames, ignore_index=True)
        LOGGER.info("Generated %d orders", len(orders))
        return orders

    def save_datasets(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        orders: pd.DataFrame,
    ) -> tuple[Path, Path, Path]:
        """Save the supplied datasets as CSV files and return their paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = (
            self.output_dir / "customers.csv",
            self.output_dir / "products.csv",
            self.output_dir / "orders.csv",
        )
        datasets = (customers, products, orders)
        for dataset, path in zip(datasets, output_paths):
            dataset.to_csv(path, index=False)
            LOGGER.info("Saved %d rows to %s", len(dataset), path)
        return output_paths


def main() -> None:
    """Generate the default datasets and save them under ``data/raw``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    generator = SyntheticDataGenerator()
    customers = generator.generate_customers()
    products = generator.generate_products()
    orders = generator.generate_orders()
    generator.save_datasets(customers, products, orders)


if __name__ == "__main__":
    main()
