import cProfile
from functools import wraps

_PROFILE = False


def set_profile(state):
    global _PROFILE
    _PROFILE = state


def profile(func):
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
            return func(*args, **kwargs)

    return wrapper
