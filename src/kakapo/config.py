"""Configuration constants and defaults for kakapo."""

from pathlib import Path

from matchms import filtering as msfilters

from kakapo.constants import DEFAULT_RULES

DEFAULT_OUTPUT_DIR = Path("kakapo_output")
DEFAULT_APPLY_FILTERS = True
DEFAULT_MIN_PEAKS = 5
DEFAULT_FRAGMENTATION_RULES = DEFAULT_RULES
DEFAULT_MIN_TOLERANCE = 0.001
DEFAULT_MASS_TOLERANCE = "0.02da"
DEFAULT_USE_CACHE = True
DEFAULT_CACHE_DIR = Path(".kakapo_cache")

MAX_TOTAL_MASS_DIFFERENCE = 300
DEFAULT_X_MODE = False
DEFAULT_SCORING_MODE = "all-peaks"

DEFAULT_FILTERS = [
    (msfilters.select_by_relative_intensity, {"intensity_from": 0.001}),
    msfilters.remove_peaks_relative_to_precursor_mz,
    (msfilters.reduce_to_number_of_peaks, {"n_max": 1000}),
    (msfilters.require_minimum_number_of_high_peaks, {"no_peaks": 5, "intensity_percent": 2.0}),
    msfilters.remove_profiled_spectra,
]
