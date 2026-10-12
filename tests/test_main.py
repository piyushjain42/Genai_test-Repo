"""Tests for the top-level Parquet pipeline script."""

from pathlib import Path

import pandas as pd
import pytest

from main import run_pipeline
from src.config import PipelineConfig


def test_run_pipeline_writes_parquet_and_reports_sizes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The pipeline should save readable Parquet files and print a summary."""
    settings = PipelineConfig(
        project_root=tmp_path,
        customer_count=5,
        product_count=3,
        order_count=10,
        random_seed=17,
    )

    output_paths = run_pipeline(settings)
    output = capsys.readouterr().out

    assert set(output_paths) == {"customers", "products", "orders"}
    for name, path in output_paths.items():
        assert path == settings.raw_data_dir / f"{name}.parquet"
        assert path.is_file()
        assert path.stat().st_size > 0
        assert not pd.read_parquet(path).empty
    assert "Generation completed in" in output
    assert "MB)" in output
