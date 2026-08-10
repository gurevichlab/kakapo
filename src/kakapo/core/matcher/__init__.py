"""Spectrum scoring engine interface."""

from kakapo.core.matcher.deduplication import _deduplicate_results
from kakapo.core.matcher.indexing import (
    ExperimentalSpectrumIndex,
    _build_structure_index,
    _filter_candidates_with_index,
    build_experimental_spectrum_index,
    get_matching_range,
)
from kakapo.core.matcher.orchestrator import match_spectra, match_spectrum

__all__ = [
    "ExperimentalSpectrumIndex",
    "_build_structure_index",
    "_filter_candidates_with_index",
    "build_experimental_spectrum_index",
    "get_matching_range",
    "match_spectra",
    "match_spectrum",
    "_deduplicate_results",
]
