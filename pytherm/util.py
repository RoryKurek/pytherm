from typing import Optional, Callable
import inspect


def args_must_be_positive(*args_to_check: list[bool]):
    def wrapper(func):
        params = inspect.signature(func).parameters
        if len(args_to_check) != len(params):
            raise ValueError(f'Number of arguments passed to args_must_be_positive does not match signature of {func.__name__}')

        def wrapped(*args, **kwargs):
            for bounds, arg in zip(args_to_check, args + list(kwargs.values())):
                if bounds is not None:
                    lb, ub = bounds
                    if lb is not None and
            return func(*args, **kwargs)
        return wrapped
    return wrapper
