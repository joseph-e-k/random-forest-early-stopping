import functools
import os

from diskcache import Cache
from diskcache.core import full_name

from .misc import function_call_to_tuple


# Set this environment variable to modify the name of the directory used for caching (useful for debugging)
CACHE_NAME_SUFFIX = os.getenv("STE_CACHE_SUFFIX", "")

# Set this environment variable to anything nonempty to turn off caching entirely
SHOULD_DUMMY_CACHING = bool(os.getenv("STE_DUMMY_CACHE", False))

# Default Cache object to be used to memoize functions if another is not specified
default_cache = Cache(os.path.join(os.path.dirname(__file__), "../.cache" + CACHE_NAME_SUFFIX))


def memoize(name=None, *, args_to_ignore=(), arg_transformations=None, cache=default_cache):
    """Return a decorator that will cause the decorated function to be memoized to disk, so it won't be computed twice for the same inputs, even across separate runs.

    Args:
        name (str, optional): Name to use in the cache. Defaults to the fully-qualified name of the decorated function. If you anticipate changing a function name but want it to keep using previously-cached results, specify this.
        args_to_ignore (Container[str], optional): Names of arguments to be excluded from the cache key. Calls that differ only in the values of these arguments will go to the same place in the cache. Defaults to ().
        arg_transformations (Mapping[str, Callable[[Any], Any]], optional): Map (indexed by name) of transformations to be applied to arguments before cache key is computed. Defaults to an empty map.
        cache (diskcache.Cache, optional): Cache object to use for memoization. Defaults to the global default_cache object in this module.

    Returns:
        Callable[[Callable[T, **P]], Callable[T, **P]]: Decorator
    """
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
