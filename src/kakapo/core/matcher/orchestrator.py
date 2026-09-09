"""Orchestration of sequential and parallel spectrum-to-structure matching workflows."""

import itertools
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

from tqdm import tqdm

from kakapo.config import DEFAULT_MIN_PEAKS, MAX_TOTAL_MASS_DIFFERENCE, MIN_SPECTRA_FOR_PARALLEL
from kakapo.constants import MAX_CHARGE, PROTON_MASS
from kakapo.core.matcher.deduplication import _deduplicate_results
from kakapo.core.matcher.hypothesis import (
    _evaluate_hypotheses,
    _generate_single_mod_hypotheses,
    _generate_two_mod_hypotheses,
    _generate_unmodified_hypotheses,
)
from kakapo.core.matcher.indexing import (
    _build_structure_index,
    _filter_candidates_with_index,
    build_experimental_spectrum_index,
)
from kakapo.models import (
    MatchResult,
    ModificationHypothesis,
    ModificationMode,
    ScoringMode,
    Spectrum,
    TheoreticalSpectrum,
    Tolerance,
)
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)

# Global variables for ProcessPoolExecutor workers
_worker_index: Any = None
_worker_theo_map: dict[str, TheoreticalSpectrum] | None = None
_worker_tolerance: Tolerance | None = None
_worker_scoring: ScoringMode | None = None
_worker_min_shared_peaks: int = DEFAULT_MIN_PEAKS
_worker_mode: ModificationMode | None = None


def _init_worker(
    index: Any,
    theo_map: dict[str, TheoreticalSpectrum],
    tolerance: Tolerance,
    scoring: ScoringMode,
    min_shared_peaks: int,
    mode: ModificationMode,
) -> None:
    """Initialize read-only shared state in process pool workers.

    Args:
        index: The _StructureIndex database search index.
        theo_map: A mapping of structure_id to TheoreticalSpectrum.
        tolerance: The tolerance window parameter.
        scoring: The scoring mode parameter.
        min_shared_peaks: The minimum number of shared peaks for filtering candidates.
        mode: The modification mode configuration.
    """
    global _worker_index
    global _worker_theo_map
    global _worker_tolerance
    global _worker_scoring
    global _worker_min_shared_peaks
    global _worker_mode
    _worker_index = index
    _worker_theo_map = theo_map
    _worker_tolerance = tolerance
    _worker_scoring = scoring
    _worker_min_shared_peaks = min_shared_peaks
    _worker_mode = mode


def _match_spectra_chunk(spectra_chunk: list[Spectrum]) -> list[MatchResult]:  # noqa: C901
    """Process a chunk of spectra inside a worker process.

    Args:
        spectra_chunk: A list of Spectrum objects to match.

    Returns:
        A list of MatchResult objects for matches that meet the thresholds.
    """
    if (
        _worker_index is None
        or _worker_theo_map is None
        or _worker_tolerance is None
        or _worker_scoring is None
        or _worker_mode is None
    ):
        raise ValueError("Worker variables not initialized.")

    results: list[MatchResult] = []
    for spec in spectra_chunk:
        spec_index = build_experimental_spectrum_index(spec)
        feasible_ids = _filter_candidates_with_index(
            spec,
            _worker_index,
            _worker_tolerance,
            min_shared_peaks=_worker_min_shared_peaks,
        )
        for struct_id in feasible_ids:
            theo = _worker_theo_map[struct_id]
            charges_to_try = (
                [spec.charge] if spec.charge is not None else list(range(1, MAX_CHARGE + 1))
            )
            for z in charges_to_try:
                precursor_neutral_mass = spec.precursor_mass * z - PROTON_MASS * z
                mass_diff = precursor_neutral_mass - theo.structure_mass
                if (
                    abs(mass_diff) <= MAX_TOTAL_MASS_DIFFERENCE
                    and max(theo.monomer_mass_vector) + mass_diff >= 0
                ):
                    spec_with_charge = replace(spec, charge=z)
                    result = match_spectrum(
                        spec_with_charge,
                        theo,
                        _worker_tolerance,
                        _worker_scoring,
                        mode=_worker_mode,
                        spec_index=spec_index,
                    )
                    if (
                        result is not None
                        and result.matched_peaks_count >= _worker_min_shared_peaks
                    ):
                        results.append(result)

    return results


def match_spectrum(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    tolerance: Tolerance,
    scoring: ScoringMode,
    mode: ModificationMode = ModificationMode.FREE_MOD,
    spec_index: Any | None = None,
) -> MatchResult | None:
    """Match an experimental spectrum to a theoretical spectrum according to mode.

    Args:
        spectrum: The experimental Spectrum instance.
        theoretical_spectrum: The TheoreticalSpectrum instance.
        tolerance: The Tolerance configuration allowed for a match.
        scoring: Scoring mode.
        mode: ModificationMode configuration.
        spec_index: Optional pre-built ExperimentalSpectrumIndex for performance.

    Returns:
        A MatchResult instance containing matching metrics, or None if no hypotheses or matches
        are found.
    """
    logger.debug(
        "Matching experimental spectrum '%s' against structure '%s' (tolerance: %s, mode: %s)",
        spectrum.spectrum_id,
        theoretical_spectrum.structure_id,
        tolerance,
        mode,
    )

    if spec_index is None:
        spec_index = build_experimental_spectrum_index(spectrum)

    precursor_neutral_mass = spectrum.get_neutral_mass()
    mass_mod = precursor_neutral_mass - theoretical_spectrum.structure_mass

    precursor_mz = spectrum.precursor_mass
    precursor_tol = tolerance.get_limit(precursor_mz)
    charge = spectrum.charge if spectrum.charge is not None else 1
    precursor_neutral_tolerance = charge * precursor_tol

    if abs(mass_mod) <= precursor_neutral_tolerance:
        hypotheses: Iterable[ModificationHypothesis] = _generate_unmodified_hypotheses(
            theoretical_spectrum
        )
    elif mode == ModificationMode.NO_MOD:
        hypotheses = ()
    elif mode == ModificationMode.FREE_MOD:
        hypotheses = _generate_single_mod_hypotheses(
            theoretical_spectrum, mass_mod, x_mode_only=False, tolerance=tolerance
        )
    elif mode == ModificationMode.X_MOD:
        hypotheses = _generate_single_mod_hypotheses(
            theoretical_spectrum, mass_mod, x_mode_only=True, tolerance=tolerance
        )
    elif mode == ModificationMode.TWO_MOD:
        hypotheses = itertools.chain(
            _generate_single_mod_hypotheses(
                theoretical_spectrum, mass_mod, x_mode_only=True, tolerance=tolerance
            ),
            _generate_two_mod_hypotheses(theoretical_spectrum, mass_mod, tolerance=tolerance),
        )
    else:
        raise ValueError(f"Unsupported modification mode: {mode}")

    return _evaluate_hypotheses(
        spec_index,
        theoretical_spectrum,
        hypotheses,
        tolerance,
        scoring,
        spectrum.spectrum_id,
        spectrum.precursor_mass,
        spectrum.charge,
    )


def match_spectra(  # noqa: C901
    spectra: Iterable[Spectrum],
    theoretical_spectra: Iterable[TheoreticalSpectrum],
    tolerance: Tolerance,
    scoring: ScoringMode,
    min_shared_peaks: int = DEFAULT_MIN_PEAKS,
    mode: ModificationMode = ModificationMode.FREE_MOD,
    num_workers: int | None = None,
) -> list[MatchResult]:
    """Match each experimental spectrum to each theoretical spectrum pairwise.

    Args:
        spectra: An iterable of experimental Spectrum instances.
        theoretical_spectra: An iterable of TheoreticalSpectrum instances.
        tolerance: The Tolerance configuration.
        scoring: Scoring mode.
        min_shared_peaks: Minimum shared peak count for candidate filtering
            (defaults to DEFAULT_MIN_PEAKS).
        mode: ModificationMode configuration.
        num_workers: Number of worker processes to use for parallel processing.

    Returns:
        A list of MatchResult instances for all pairwise matches.
    """
    logger.debug("Starting batch matching (cross-matching spectra)...")
    spectra_list = list(spectra)
    theo_list = [theo for theo in theoretical_spectra if theo.monomer_count > 1]
    logger.debug(
        "Loaded %d experimental spectra and %d theoretical spectra for matching",
        len(spectra_list),
        len(theo_list),
    )

    index = _build_structure_index(theo_list)
    theo_map = {theo.structure_id: theo for theo in theo_list}

    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    # Fallback to sequential execution if input size is small or only 1 worker is available
    if len(spectra_list) <= MIN_SPECTRA_FOR_PARALLEL or num_workers <= 1:
        logger.debug("Running matching sequentially.")
        results: list[MatchResult] = []
        for spec in tqdm(spectra_list, desc="Matching spectra", unit="spec"):
            spec_index = build_experimental_spectrum_index(spec)
            feasible_ids = _filter_candidates_with_index(
                spec, index, tolerance, min_shared_peaks=min_shared_peaks
            )
            logger.debug(
                "Found %d feasible structures for spectrum '%s' (min shared peaks: %d)",
                len(feasible_ids),
                spec.spectrum_id,
                min_shared_peaks,
            )
            for struct_id in feasible_ids:
                theo = theo_map[struct_id]
                charges_to_try = (
                    [spec.charge] if spec.charge is not None else list(range(1, MAX_CHARGE + 1))
                )
                for z in charges_to_try:
                    precursor_neutral_mass = spec.precursor_mass * z - PROTON_MASS * z
                    mass_diff = precursor_neutral_mass - theo.structure_mass
                    if (
                        abs(mass_diff) <= MAX_TOTAL_MASS_DIFFERENCE
                        and max(theo.monomer_mass_vector) + mass_diff >= 0
                    ):
                        spec_with_charge = replace(spec, charge=z)
                        result = match_spectrum(
                            spec_with_charge,
                            theo,
                            tolerance,
                            scoring,
                            mode=mode,
                            spec_index=spec_index,
                        )
                        if result is not None and result.matched_peaks_count >= min_shared_peaks:
                            results.append(result)
    else:
        logger.debug("Running matching in parallel using %d workers", num_workers)
        # Determine chunk size to balance task distribution and scheduling overhead
        chunk_size = max(1, len(spectra_list) // (num_workers * 4))
        chunks = [spectra_list[i : i + chunk_size] for i in range(0, len(spectra_list), chunk_size)]

        results = []
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(index, theo_map, tolerance, scoring, min_shared_peaks, mode),
        ) as executor:
            # Using executor.map distributes chunk tasks to worker processes
            for chunk_res in executor.map(_match_spectra_chunk, chunks):
                results.extend(chunk_res)

    logger.debug(
        "Completed batch matching. Generated %d total MatchResult(s).",
        len(results),
    )
    return _deduplicate_results(results)
