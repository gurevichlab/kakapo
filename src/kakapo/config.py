"""Configuration constants and defaults for kakapo."""

from pathlib import Path

from kakapo.constants import DEFAULT_RULES

DEFAULT_OUTPUT_DIR = Path("kakapo_output")
DEFAULT_APPLY_FILTERS = True
DEFAULT_MIN_PEAKS = 5
DEFAULT_FRAGMENTATION_RULES = DEFAULT_RULES
DEFAULT_MASS_TOLERANCE = 0.02
DEFAULT_USE_CACHE = True
DEFAULT_CACHE_DIR = Path(".kakapo_cache")

MAX_TOTAL_MASS_DIFFERENCE = 300
