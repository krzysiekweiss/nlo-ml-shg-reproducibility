"""Format progress messages for sequential pipeline operations."""


from time import perf_counter

def step_header(step_id: int, title: str):
    print(f"\n[STEP {step_id}] {title}")

def kv(key: str, value):
    print(f"  → {key:<16}: {value}")

def done(stats: dict | None = None):
    print("  ✓ done")
    if stats:
        for k, v in stats.items():
            print(f"    {k:<18}: {v}")

class StepTimer:
    def __enter__(self):
        self.t0 = perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = perf_counter() - self.t0
