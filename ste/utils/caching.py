import functools
import inspect
import os

from diskcache import Cache
from diskcache.core import full_name


default_cache = Cache(os.path.join(os.path.dirname(__file__), "../.cache"))


def _robust_cache_key(function, name, args_to_ignore, arg_transformations, *args, **kwargs):
    signature = inspect.signature(function)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    key = [name]

    for arg_name, arg_value in bound_arguments.arguments.items():
        if arg_name in args_to_ignore:
            continue
        if arg_transformations is not None and arg_name in arg_transformations:
            transformation = arg_transformations[arg_name]
            arg_value = transformation(arg_value)
        key.append((arg_name, arg_value))

    return tuple(key)


def memoize(name=None, *, args_to_ignore=(), arg_transformations=None, cache=default_cache):
    def decorator(function, name=name, args_to_ignore=args_to_ignore):
        if name is None:
            name = full_name(function)
        memoized = cache.memoize(name=name)(function)
        memoized.__cache_key__ = functools.partial(_robust_cache_key, memoized, name, args_to_ignore, arg_transformations)
        return memoized
    return decorator