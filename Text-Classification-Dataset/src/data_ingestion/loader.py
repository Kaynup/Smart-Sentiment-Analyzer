import os
import pandas as pd
from typing import Iterable, List
from . import config

def stream_texts(
    filepaths: List[str],
    text_col: str = config.TEXT_COLUMN,
    chunksize: int = config.CHUNK_SIZE
) -> Iterable[str]:
    """
    Yield text rows from multiple CSVs, chunk by chunk.

    Reads CSVs safely with:
    - bad lines skipped
    - custom separator '~'
    - quote character '"'
    """
    for fp in filepaths:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"CSV file not found: {fp}")
        for chunk in pd.read_csv(
            fp,
            usecols=[text_col],
            chunksize=chunksize,
            on_bad_lines='skip',
            sep='~',
            quotechar='"'
        ):
            for val in chunk[text_col].dropna().astype(str).tolist():
                yield val
