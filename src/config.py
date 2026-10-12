"""Project paths and default settings for the e-commerce data pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Store project directories and synthetic-data generation defaults."""

    project_root: Path = Path(__file__).resolve().parent.parent
    customer_count: int = 100_000
    product_count: int = 10_000
    order_count: int = 1_000_000
    random_seed: int = 42

    @property
    def raw_data_dir(self) -> Path:
        """Return the directory used to store generated CSV files."""
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Return the directory used to store Spark analysis results."""
        return self.project_root / "data" / "processed"


DEFAULT_CONFIG = PipelineConfig()
