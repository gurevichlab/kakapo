"""Configuration constants and defaults for kakapo."""

import os
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path("kakapo_output")
DEFAULT_LOG_DIR = Path(os.path.join(DEFAULT_OUTPUT_DIR, "logs"))
DEFAULT_LOG_PATH = Path(os.path.join(DEFAULT_LOG_DIR, "kakapo.log"))


