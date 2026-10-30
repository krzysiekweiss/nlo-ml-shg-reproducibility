"""Record notebook output with optional elapsed-time information."""


import sys
from pathlib import Path
from datetime import datetime

try:
    from IPython import get_ipython
except ImportError:
    get_ipython = None


class Tee:
    """Duplicate output to a notebook stream and a file stream (Jupyter-safe)."""

    def __init__(self, notebook_stream, file_stream):
        self.notebook_stream = notebook_stream
        self.file_stream = file_stream

    def write(self, data):
        if self.notebook_stream is not None:
            self.notebook_stream.write(data)
            self.notebook_stream.flush()
        self.file_stream.write(data)
        self.file_stream.flush()

    def flush(self):
        if self.notebook_stream is not None:
            self.notebook_stream.flush()
        self.file_stream.flush()

    def isatty(self):
        stream = self.notebook_stream
        if stream is not None and hasattr(stream, "isatty"):
            return bool(stream.isatty())
        return False

    def set_parent(self, parent):
        """
        Forward IPykernel parent-header updates so outputs appear
        under the currently executing notebook cell.
        """
        if self.notebook_stream is not None and hasattr(self.notebook_stream, "set_parent"):
            self.notebook_stream.set_parent(parent)

    @property
    def encoding(self):
        return getattr(self.notebook_stream, "encoding", "utf-8")


class NotebookLogger:
    """
    Jupyter-safe logger that duplicates stdout/stderr to a file
    while keeping normal notebook output visible.
    """

    def __init__(
        self,
        project_dir: str | Path,
        prefix: str = "notebook",
        timestamp_fmt: str = "%Y%m%d_%H%M%S",
    ):
        self.project_dir = Path(project_dir).resolve()
        self.log_dir = self.project_dir / "logs"
        self.prefix = prefix
        self.timestamp_fmt = timestamp_fmt

        self.log_file: Path | None = None
        self._fh = None
        self._old_stdout = None
        self._old_stderr = None

    def _get_notebook_stream(self):
        """Return the IPython stdout stream if running in a notebook."""
        if get_ipython is None:
            return sys.stdout
        ip = get_ipython()
        if ip is None:
            return sys.stdout
        return sys.stdout  # IPython replaces sys.stdout with its own stream

    def start(self, header_lines: list[str] | None = None) -> Path:
        """Start tee-logging."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime(self.timestamp_fmt)
        self.log_file = self.log_dir / f"{self.prefix}_{ts}.log"
        self._fh = open(self.log_file, "w", encoding="utf-8")

        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr

        sys.stdout = Tee(self._old_stdout, self._fh)
        sys.stderr = Tee(self._old_stderr, self._fh)

        if header_lines:
            print("=" * 80)
            for line in header_lines:
                print(line)
            print("=" * 80)

        print("Logging started.")
        print("Log file:", self.log_file.resolve())

        return self.log_file

    def stop(self) -> None:
        """Restore original stdout/stderr."""
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
        if self._old_stderr is not None:
            sys.stderr = self._old_stderr

        if self._fh is not None:
            self._fh.close()

        self._fh = None
