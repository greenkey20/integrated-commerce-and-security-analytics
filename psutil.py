"""Minimal psutil stub used only for tests collection in environments
where the real psutil package is not installed.

This provides just enough of the public API used by the project's tests:
- psutil.Process().memory_info().rss
- psutil.virtual_memory() -> object with total/available/percent

This is intentionally lightweight and approximate; it should only be used
for CI/local tests runs when the real psutil isn't available.
"""
from types import SimpleNamespace
import resource


class Process:
    def __init__(self, pid=None):
        self.pid = pid

    def memory_info(self):
        # Try to return a reasonable RSS value using resource; fallback to 50MB
        try:
            # ru_maxrss may be in bytes (macOS) or kilobytes (Linux); normalize conservatively
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if ru > 10 * 1024 * 1024:  # looks like bytes
                rss = int(ru)
            else:
                rss = int(ru) * 1024
        except Exception:
            rss = 50 * 1024 * 1024
        return SimpleNamespace(rss=rss)


def virtual_memory():
    # Provide simple static totals (8GB total, 6GB available) and a percent used
    total = 8 * 1024 ** 3
    available = 6 * 1024 ** 3
    used = total - available
    percent = (used / total) * 100
    return SimpleNamespace(total=total, available=available, percent=percent)


# compatibility: top-level Process callable
def Process_class(pid=None):
    return Process(pid)

# expose Process symbol the tests use
Process = Process

