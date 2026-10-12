"""Tests for synthetic e-commerce data generation."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_generator import (
    CUSTOMER_COLUMNS,
    ORDER_COLUMNS,
    PRODUCT_CATEGORIES,
    PRODUCT_COLUMNS,
    SyntheticDataGenerator,
)


def test_generate_datasets_have_expected_fields_and_valid_relationships() -> None:
    """Generated frames should have the requested schema and valid order IDs."""
    generator = SyntheticDataGenerator(
        customer_count=20,
        product_count=8,
        order_count=100,
        seed=7,
    )
    customers = generator.generate_customers()
    products = generator.generate_products()
    orders = generator.generate_orders()

    assert isinstance(customers, pd.DataFrame)
    assert isinstance(products, pd.DataFrame)
    assert isinstance(orders, pd.DataFrame)
    assert tuple(customers.columns) == CUSTOMER_COLUMNS
    assert tuple(products.columns) == PRODUCT_COLUMNS
    assert tuple(orders.columns) == ORDER_COLUMNS
    assert set(orders["customer_id"]) <= set(customers["customer_id"])
    assert set(orders["product_id"]) <= set(products["product_id"])
    assert customers["age"].between(18, 90).all()
    assert set(products["category"]) <= set(PRODUCT_CATEGORIES)
    assert products["price"].between(10, 500).all()
    assert (products["stock"] >= 0).all()
    assert products["rating"].between(1, 5).all()
    assert orders["quantity"].between(1, 10).all()


def test_order_frequency_is_pareto_like() -> None:
    """A small group of customers should account for most generated orders."""
    generator = SyntheticDataGenerator(
        customer_count=1_000,
        product_count=5,
        order_count=20_000,
        seed=23,
    )
    generator.generate_customers()
    generator.generate_products()
    orders = generator.generate_orders()

    counts = orders["customer_id"].value_counts()
    top_fifth_order_share = counts.nlargest(200).sum() / len(orders)
    assert 0.65 <= top_fifth_order_share <= 0.82


def test_orders_require_customers_and_products() -> None:
    """Orders cannot be generated until their referenced datasets exist."""
    generator = SyntheticDataGenerator(customer_count=2, product_count=2, order_count=2)

    with pytest.raises(RuntimeError, match="Generate customers and products"):
        generator.generate_orders()


def test_save_datasets_creates_output_directory(tmp_path: Path) -> None:
    """Saving generated frames should create the target folder and CSV files."""
    generator = SyntheticDataGenerator(
        customer_count=2,
        product_count=2,
        order_count=2,
        output_dir=tmp_path / "raw",
    )
    customers = generator.generate_customers()
    products = generator.generate_products()
    orders = generator.generate_orders()

    paths = generator.save_datasets(customers, products, orders)

    assert all(path.is_file() for path in paths)
    assert [path.name for path in paths] == ["customers.csv", "products.csv", "orders.csv"]
