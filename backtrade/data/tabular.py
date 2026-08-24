from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather
import pyarrow.parquet as pq


SUPPORTED_TABLE_EXTENSIONS = (".parquet", ".csv", ".csv.gz", ".feather")


def table_format(path: str | Path) -> str:
    name = Path(path).name.lower()
    if name.endswith(".csv.gz"):
        return "csv"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".feather"):
        return "feather"
    if name.endswith(".csv"):
        return "csv"
    raise ValueError(
        f"unsupported tabular input format: {path}; "
        f"supported extensions are {', '.join(SUPPORTED_TABLE_EXTENSIONS)}"
    )


def _columns_arg(columns: Sequence[str] | None) -> list[str] | None:
    return list(columns) if columns is not None else None


def table_columns(path: str | Path) -> list[str]:
    table_path = Path(path).expanduser()
    fmt = table_format(table_path)
    if fmt == "parquet":
        return list(pq.ParquetFile(table_path).schema.names)
    if fmt == "feather":
        return list(feather.read_table(table_path).schema.names)
    return [str(column) for column in pd.read_csv(table_path, nrows=0).columns]


def read_table(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    table_path = Path(path).expanduser()
    fmt = table_format(table_path)
    selected = _columns_arg(columns)
    if fmt == "parquet":
        return pd.read_parquet(table_path, columns=selected)
    if fmt == "feather":
        return pd.read_feather(table_path, columns=selected)
    return pd.read_csv(table_path, usecols=selected)


def table_row_count(path: str | Path) -> int:
    table_path = Path(path).expanduser()
    fmt = table_format(table_path)
    if fmt == "parquet":
        return int(pq.ParquetFile(table_path).metadata.num_rows)
    if fmt == "feather":
        return int(feather.read_table(table_path).num_rows)
    columns = table_columns(table_path)
    if not columns:
        return 0
    return int(len(pd.read_csv(table_path, usecols=[columns[0]])))


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    table_path = Path(path).expanduser()
    fmt = table_format(table_path)
    if fmt == "parquet":
        frame.to_parquet(table_path, index=False)
    elif fmt == "feather":
        frame.reset_index(drop=True).to_feather(table_path)
    else:
        compression = "gzip" if table_path.name.lower().endswith(".csv.gz") else None
        frame.to_csv(table_path, index=False, compression=compression)


def resolve_table_candidate(root: str | Path, stem: str) -> Path:
    root_path = Path(root).expanduser()
    candidates = [
        root_path / f"{stem}{extension}"
        for extension in SUPPORTED_TABLE_EXTENSIONS
        if (root_path / f"{stem}{extension}").is_file()
    ]
    if len(candidates) > 1:
        raise ValueError(f"multiple tabular inputs match {stem}: {candidates}")
    return candidates[0] if candidates else root_path / f"{stem}.parquet"


__all__ = [
    "SUPPORTED_TABLE_EXTENSIONS",
    "read_table",
    "resolve_table_candidate",
    "table_columns",
    "table_format",
    "table_row_count",
    "write_table",
]
