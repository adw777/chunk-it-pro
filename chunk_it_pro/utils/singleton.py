import threading
from weakref import WeakKeyDictionary


class Singleton(type):
    """
    Thread-safe Singleton metaclass.

    Ensures that only one instance of each class using this metaclass
    is ever created, even under concurrent access. Uses a lock to
    guard initialization and a WeakKeyDictionary to avoid pinning
    instances forever if classes are dynamically created/destroyed.
    """
    _instances = WeakKeyDictionary()
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls] 