"""
tqdm_test.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from multiprocessing import Manager

from time import sleep

manager = Manager()
MESSAGES = manager.Queue()
TQDM_COUNT = manager.Value('i', 0)
TQDM_COUNT_LOCK = manager.Lock()


def next_id():
    with TQDM_COUNT_LOCK:
        id_val = TQDM_COUNT.get() + 1
        TQDM_COUNT.set(id_val)
        return id_val


from functools import wraps
import inspect


class tqdm_:

    def __init__(self, *args, **kwargs):
        # assert 'position' not in kwargs
        #
        # MESSAGES

        # super().__init__(*args, **kwargs)
        self.obj = tqdm(*args, **kwargs)

    def __get_dummy__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with TQDM_COUNT_LOCK:
                return func(*args, **kwargs)

        return wrapper

    def __getattr__(self, item):
        # value = super().__getattr__(item)
        if item == 'obj' or item == '__get_dummy__':
            return object.__getattribute__(self, item)

        value = getattr(self.obj, item)

        print(inspect.ismethod(value),
              inspect.isfunction(value))
        if inspect.ismethod(value):
            return self.__get_dummy__(value)

        return value

    def __iter__(self):
        return self.obj.__iter__()

    def __setattr__(self, key, value):
        if key == 'obj':
            object.__setattr__(self, key, value)
        else:
            setattr(self.obj, key, value)


def job(i):
    # pbar = tqdm_(total=50, desc=str(i), position=i)

    # MESSAGES.put(i)
    # for _ in range(50):
    for _ in tqdm_(range(50), desc=str(i), position=i, leave=True):
        # pbar.update()
        sleep(.1)
    # pbar.refresh()


def run():
    n_jobs = 6
    _ = Parallel(n_jobs=-1)(
        delayed(job)(i)
        for i in range(n_jobs)
    )
    print('\n' * n_jobs)
    while not MESSAGES.empty():
        print(MESSAGES.get())


run()
