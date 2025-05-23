import time

import numpy as np

FUNCTIONS = {
    "add": lambda x, y: x + y,
    "sleep": time.sleep,
    "round": np.round
}


def invoke(func, *args, **kwargs):
    return FUNCTIONS[func](*args, **kwargs)
