"""Record elapsed time for pipeline operations."""


import time

class LogTiming:
    """
    Lightweight timing helper for measuring and printing execution time
    of a code block or function.
    """

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self._t0 = time.perf_counter() if enabled else None
        self._stopped = False

    def __enter__(self):
        # Handle case where object is created with enabled=True but _t0=None
        if self.enabled and self._t0 is None:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Always stop the timer when exiting the context block
        self.stop()
        return False  # Do not suppress exceptions

    def stop(self):
        """Stop timing and print elapsed time."""
        if not self.enabled or self._stopped:
            return

        dt = time.perf_counter() - self._t0
        self._stopped = True

        if dt < 60:
            print(f"[TIME] {self.name}: {dt:.2f} s")
        elif dt < 3600:
            m = int(dt // 60)
            s = dt % 60
            print(f"[TIME] {self.name}: {m:d} min {s:.1f} s")
        else:
            h = int(dt // 3600)
            m = int((dt % 3600) // 60)
            s = dt % 60
            print(f"[TIME] {self.name}: {h:d} h {m:d} min {s:.0f} s")

    def __del__(self):
        """Ensure timing is printed if stop() was not called explicitly."""
        self.stop()
