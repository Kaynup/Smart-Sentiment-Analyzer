import time
from typing import Iterable, List

def batch_iterator(iterable, batch_size: int):
    """Yield successive batches from iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def log_time(func):
    """Decorator to measure execution time of functions."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱ {func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper
