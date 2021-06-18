"""Bounded local execution context helpers."""
from contextlib import contextmanager


def validate_threads(count):
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("CPU thread count must be in [1,256]")
    return count


@contextmanager
def cpu_threads(count):
    import torch
    validate_threads(count)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(count)
        yield
    finally:
        torch.set_num_threads(previous)
