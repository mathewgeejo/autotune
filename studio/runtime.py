"""Long-running inference survives automatic sleep without changing power plans."""
from contextlib import contextmanager
import os
import time


def active_time():
    """Monotonic seconds excluding Windows standby/hibernation."""
    if os.name == 'nt':
        import ctypes
        value = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)):
            return value.value/10_000_000
    return time.monotonic()


@contextmanager
def keep_awake():
    """Inhibit automatic idle sleep only while this thread is doing work.

    The display may turn off; explicit user sleep/lid behavior is respected.
    Windows releases the request if the process or thread is terminated.
    """
    previous = 0
    if os.name == 'nt':
        import ctypes
        function = ctypes.windll.kernel32.SetThreadExecutionState
        function.argtypes = [ctypes.c_uint]
        function.restype = ctypes.c_uint
        previous = function(0x80000001)  # CONTINUOUS | SYSTEM_REQUIRED
    try:
        yield
    finally:
        if os.name == 'nt' and previous:
            function(previous)
