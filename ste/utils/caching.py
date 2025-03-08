import functools
import inspect
import os

from diskcache import Cache
from diskcache.core import full_name

from .misc import function_call_to_tuple


CACHE_NAME_SUFFIX = os.getenv("STE_CACHE_SUFFIX", "")
default_cache = Cache(os.path.join(os.path.dirname(__file__), "../.cache" + CACHE_NAME_SUFFIX))
SHOULD_DUMMY_CACHING = bool(os.getenv("STE_DUMMY_CACHE", False))


def memoize(name=None, *, args_to_ignore=(), arg_transformations=None, cache=default_cache):
    arg_transformations = arg_transformations or {}
    def decorator(function, name=name, args_to_ignore=args_to_ignore):
        if name is None:
            name = full_name(function)
        memoized = cache.memoize(name=name)(function)
        memoized.__cache_key__ = functools.partial(function_call_to_tuple, memoized, name, args_to_ignore, arg_transformations)
        return memoized
    return decorator


if SHOULD_DUMMY_CACHING:
    memoize = lambda *args, **kwargs: (lambda function: function)
