import functools
import inspect
import os

from diskcache import Cache
from diskcache.core import full_name


cache = Cache(os.path.join(os.path.dirname(__file__), "../.cache"))


def _robust_cache_key(function, name, args_to_ignore, *args, **kwargs):
    signature = inspect.signature(function)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    return (name,) + tuple(
        (key, value) for (key, value) in bound_arguments.arguments.items() if key not in args_to_ignore
    )


def memoize(name=None, args_to_ignore=()):
    def decorator(function, name=name, args_to_ignore=args_to_ignore):
        if name is None:
            name = full_name(function)
        memoized = cache.memoize(name=name)(function)
        memoized.__cache_key__ = functools.partial(_robust_cache_key, memoized, name, args_to_ignore)
        return memoized
    return decorator