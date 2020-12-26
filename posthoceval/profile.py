import cProfile
from functools import wraps
from memory_profiler import profile as _mem_profile

_PROFILE = False


def set_profile(state):
    global _PROFILE
    _PROFILE = state


def profile(func):
    if _PROFILE:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _PROFILE:
                profiler = cProfile.Profile()
                try:
                    profiler.enable()
                    ret = func(*args, **kwargs)
                    profiler.disable()
                    return ret
                finally:
                    filename = func.__name__ + '.pstat'
                    profiler.dump_stats(filename)
    else:
        wrapper = func

    return wrapper


def mem_profile(func=None, *args, **kwargs):
    if _PROFILE:
        return _mem_profile(func, *args, **kwargs)
    elif func is None:
        def wrapper(func_):
            return mem_profile(func_)
    else:
        @wraps(func)
        def wrapper(*args_, **kwargs_):
            return func(*args_, **kwargs_)

    return wrapper
