"""Dataset-specific loaders. Each converts one native format to the unified schema.

Importing this package registers every loader, so `get_loader("nq")` works
without the caller knowing which module defines it. Without these imports the
`@register_loader` decorators never run and the registry is empty.
"""
from src.data.loaders.base import (
    DatasetFormatError,
    DatasetLoader,
    LoadResult,
    available_loaders,
    get_loader,
    register_loader,
)
from src.data.loaders.hotpotqa import HotpotQaLoader
from src.data.loaders.natural_questions import NaturalQuestionsLoader
from src.data.loaders.nq_parquet import NaturalQuestionsParquetLoader
from src.data.loaders.qasper import QasperLoader

__all__ = [
    "DatasetFormatError",
    "DatasetLoader",
    "HotpotQaLoader",
    "LoadResult",
    "NaturalQuestionsLoader",
    "NaturalQuestionsParquetLoader",
    "QasperLoader",
    "available_loaders",
    "get_loader",
    "register_loader",
]
