"""Configure artifact and CIF locations from environment variables."""
import os
from pathlib import Path

DATA_DIR = str(Path(os.environ.get("NLO_DATA_DIR", str(Path.home() / "data"))).expanduser())
PROJECTS_BASE = str(Path(os.environ.get("NLO_PROJECTS_BASE", str(Path(DATA_DIR) / "NLO_ML"))).expanduser())
COD_DIR = str(Path(DATA_DIR) / "COD" / "unpacked" / "cif")
MP_DIR = str(Path(DATA_DIR) / "MP" / "cif")
TEST_DIR = str(Path(DATA_DIR) / "test")
