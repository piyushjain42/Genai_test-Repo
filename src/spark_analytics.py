"""PySpark analytics for synthetic e-commerce sales data."""

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import DEFAULT_CONFIG, PipelineConfig

LOGGER = logging.getLogger(__name__)


class SalesAnalytics:
    """Load Parquet datasets and calculate e-commerce sales insights."""

    def __init__(self, spark: SparkSession | None = None) -> None:
        """Initialize analytics, optionally with an existing Spark session."""
        self.spark = spark

    def create_spark_session(self) -> SparkSession:
        """Create a local Spark session with 4 GB memory, AQE, and Kryo."""
        self.spark = (
            SparkSession.builder.appName("EcommerceSalesAnalytics")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        LOGGER.info("Created local Spark session with adaptive execution and Kryo.")
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet dataset using this instance's Spark session."""
        if self.spark is None:
            raise RuntimeError(
                "Create a Spark session before loading Parquet data."
            )

        LOGGER.info("Loading Parquet dataset from %s", path)
        return self.spark.read.parquet(str(path))

    @staticmethod
    def _orders_with_revenue(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join orders to product prices and calculate line-item revenue."""
        return orders_df.join(products_df, on="product_id", how="inner").withColumn(
            "revenue",
            F.col("price").cast("double") * F.col("quantity").cast("double"),
        )

    def top_customers_by_revenue(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
        n: int = 10,
    ) -> DataFrame:
        """Return the top ``n`` customers ranked by total order revenue."""
        if n <= 0:
            raise ValueError("n must be a positive integer.")

        return (
            self._orders_with_revenue(orders_df, products_df)
            .groupBy("customer_id")
            .agg(F.round(F.sum("revenue"), 2).alias("total_revenue"))
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )

    def sales_by_category(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Aggregate total revenue and units sold for each product category."""
        return (
            self._orders_with_revenue(orders_df, products_df)
            .groupBy("category")
            .agg(
                F.round(F.sum("revenue"), 2).alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )

    def monthly_trends(
        self,
        orders_df: DataFrame,
        products_df: DataFrame,
    ) -> DataFrame:
        """Return monthly revenue with month-over-month growth as a percentage.

        The first month has no previous month, so its growth value is null.
        """
        monthly_revenue = (
            self._orders_with_revenue(orders_df, products_df)
            .withColumn(
                "month",
                F.date_format(F.to_date(F.col("order_date")), "yyyy-MM"),
            )
            .groupBy("month")
            .agg(F.round(F.sum("revenue"), 2).alias("revenue"))
        )
        month_window = Window.orderBy("month")
        previous_month_revenue = F.lag("revenue").over(month_window)

        return (
            monthly_revenue.withColumn(
                "revenue_growth_pct",
                F.when(
                    previous_month_revenue.isNotNull()
                    & (previous_month_revenue != 0),
                    F.round(
                        (F.col("revenue") - previous_month_revenue)
                        / previous_month_revenue
                        * 100,
                        2,
                    ),
                ).otherwise(F.lit(None).cast("double")),
            )
            .orderBy("month")
        )


def build_insights(spark: SparkSession, raw_data_dir: Path) -> dict[str, DataFrame]:
    """Load raw Parquet datasets and calculate all sales insight datasets."""
    analytics = SalesAnalytics(spark)
    orders_path = raw_data_dir / "orders.parquet"
    products_path = raw_data_dir / "products.parquet"
    for input_path in (orders_path, products_path):
        if not input_path.exists():
            raise FileNotFoundError(f"Required input dataset was not found: {input_path}")

    orders = analytics.load_parquet(orders_path)
    products = analytics.load_parquet(products_path)
    return {
        "top_customers": analytics.top_customers_by_revenue(orders, products),
        "category_revenue": analytics.sales_by_category(orders, products),
        "monthly_sales": analytics.monthly_trends(orders, products),
    }


def write_insights(insights: dict[str, DataFrame], output_dir: Path) -> None:
    """Write each insight as a CSV dataset in its own output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, dataframe in insights.items():
        destination = output_dir / name
        dataframe.coalesce(1).write.mode("overwrite").option("header", True).csv(
            str(destination)
        )
        LOGGER.info("Wrote %s insights to %s", name, destination)


def run_analytics(config: PipelineConfig = DEFAULT_CONFIG) -> None:
    """Run the sales analyses and persist their results."""
    analytics = SalesAnalytics()
    spark = analytics.create_spark_session()
    try:
        results = build_insights(spark, config.raw_data_dir)
        write_insights(results, config.processed_data_dir)
        LOGGER.info("Completed analytics for %d insight datasets", len(results))
    finally:
        spark.stop()


def main() -> None:
    """Run the configured Spark sales analytics job."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_analytics()


if __name__ == "__main__":
    main()
