"""Tests for the analytics runner's orchestration and cleanup."""

from pathlib import Path

import pytest

from run_analytics import run_analytics


class FakeDataFrame:
    """Minimal DataFrame stand-in for runner tests."""

    def __init__(self) -> None:
        self.displayed = False

    def show(self) -> None:
        """Record that the runner displayed this result."""
        self.displayed = True


class FakeSparkSession:
    """Minimal Spark session stand-in for runner tests."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        """Record that the runner stopped this session."""
        self.stopped = True


class FakeSalesAnalytics:
    """Stand-in analytics class that records loaded paths and results."""

    instance: "FakeSalesAnalytics | None" = None

    def __init__(self) -> None:
        self.session = FakeSparkSession()
        self.loaded_paths: list[Path] = []
        self.results: list[FakeDataFrame] = []
        self.analytics_inputs: list[tuple[FakeDataFrame, FakeDataFrame]] = []
        FakeSalesAnalytics.instance = self

    def create_spark_session(self) -> FakeSparkSession:
        """Return the test Spark session."""
        return self.session

    def load_parquet(self, path: Path) -> FakeDataFrame:
        """Record a dataset path and return a stand-in DataFrame."""
        self.loaded_paths.append(path)
        return FakeDataFrame()

    def top_customers_by_revenue(
        self,
        orders_df: FakeDataFrame,
        products_df: FakeDataFrame,
    ) -> FakeDataFrame:
        """Return a stand-in top-customer result."""
        return self._new_result(orders_df, products_df)

    def sales_by_category(
        self,
        orders_df: FakeDataFrame,
        products_df: FakeDataFrame,
    ) -> FakeDataFrame:
        """Return a stand-in category result."""
        return self._new_result(orders_df, products_df)

    def monthly_trends(
        self,
        orders_df: FakeDataFrame,
        products_df: FakeDataFrame,
    ) -> FakeDataFrame:
        """Return a stand-in monthly-trends result."""
        return self._new_result(orders_df, products_df)

    def _new_result(
        self,
        orders_df: FakeDataFrame,
        products_df: FakeDataFrame,
    ) -> FakeDataFrame:
        """Create and record a stand-in analytics result."""
        self.analytics_inputs.append((orders_df, products_df))
        result = FakeDataFrame()
        self.results.append(result)
        return result


def test_runner_loads_all_data_runs_analyses_and_stops_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should load required inputs, display results, and clean up."""
    monkeypatch.setattr("run_analytics.SalesAnalytics", FakeSalesAnalytics)

    run_analytics(tmp_path)

    instance = FakeSalesAnalytics.instance
    assert instance is not None
    assert instance.loaded_paths == [
        tmp_path / "customers.parquet",
        tmp_path / "products.parquet",
        tmp_path / "orders.parquet",
    ]
    assert len(instance.results) == 3
    assert all(result.displayed for result in instance.results)
    assert len(instance.analytics_inputs) == 3
    assert instance.session.stopped
