"""Configuration constants and defaults for kakapo."""

from pathlib import Path

from kakapo.constants import DEFAULT_RULES
from kakapo.utils.filters import (
    keep_highest_intensity_in_radius,
    reduce_to_number_of_peaks,
    remove_peaks_relative_to_precursor_mz_charge_aware,
    require_minimum_number_of_high_peaks,
    select_by_relative_intensity,
)

DEFAULT_OUTPUT_DIR = Path("kakapo_output")
DEFAULT_APPLY_FILTERS = True
DEFAULT_MIN_PEAKS = 5
DEFAULT_FRAGMENTATION_RULES = DEFAULT_RULES
DEFAULT_MIN_TOLERANCE = 0.001
DEFAULT_MASS_TOLERANCE = "0.02da"
DEFAULT_USE_CACHE = True
DEFAULT_CACHE_DIR = Path(".kakapo_cache")
DEFAULT_MAX_HITS_PER_SPECTRUM = 10

MAX_TOTAL_MASS_DIFFERENCE = 250
DEFAULT_MODIFICATION_MODE = "free-mod"
DEFAULT_SCORING_MODE = "exp-peaks"
MIN_SPECTRA_FOR_PARALLEL = 5

DEFAULT_FILTERS = [
    (select_by_relative_intensity, {"intensity_from": 0.001}),
    remove_peaks_relative_to_precursor_mz_charge_aware,
    (keep_highest_intensity_in_radius, {"radius": 0.025}),
    (reduce_to_number_of_peaks, {"n_max": 1000}),
    (require_minimum_number_of_high_peaks, {"no_peaks": 5, "intensity_percent": 2.0}),
]
