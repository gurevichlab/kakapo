"""Scoring algorithms for matching theoretical and experimental mass spectra peaks."""

import bisect

from kakapo.constants import SCORING_WEIGHTS
from kakapo.core.matcher.indexing import ExperimentalSpectrumIndex
from kakapo.core.matcher.indexing import ExperimentalSpectrumIndex, get_left_index
from kakapo.models import (
    MatchedPeak,
    ScoringMode,
    TheoreticalIonWithMass,
    Tolerance,
)
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def _match_peaks(
    spec_index: ExperimentalSpectrumIndex,
    theory_ions_with_mass: list[TheoreticalIonWithMass],
    max_charge: int,
    tolerance: Tolerance,
) -> tuple[int, list[MatchedPeak]]:
    """Greedy peak matching of pre-calculated theoretical ions against experimental peaks.

    Args:
        spec_index: Pre-indexed experimental spectrum.
        theory_ions_with_mass: Sorted list of theoretical ions with pre-calculated masses.
        max_charge: Maximum charge to consider.
        tolerance: Tolerance configuration.

    Returns:
        Tuple of (matched_peaks_count, matched_peaks).
    """
    matched_peaks_count = 0
    matched_peaks: list[MatchedPeak] = []
    exp_mzs = spec_index.mzs_list
    n_exp = len(exp_mzs)

    exp_idx = 0

    for ion in theory_ions_with_mass:
        mz = ion.mz
        charge = ion.charge

        if charge > max_charge:
            continue

        tol = tolerance.get_limit(mz)
        min_mass = mz - tol
        max_mass = mz + tol

        left_idx = bisect.bisect_left(exp_mzs, min_mass, lo=exp_idx)
        left_idx = get_left_index(spec_index, min_mass)
        cand_idx = max(exp_idx, left_idx)

        if cand_idx < n_exp and exp_mzs[cand_idx] <= max_mass:
            exp_mz = exp_mzs[cand_idx]
            matched_peaks_count += 1
            matched_peaks.append(
                MatchedPeak(
                    theoretical_mz=mz,
                    experimental_mz=exp_mz,
                    experimental_peak_intensity=float(spec_index.intensities[cand_idx]),
                    experimental_peak_idx=cand_idx,
                    charge=charge,
                    theoretical_fragment_mask=ion.mask,
                )
            )
            logger.debug(
                "Matched ion '%s' (m/z: %f, charge: %d) to peak (m/z: %f)",
                ion.fragment_id,
                mz,
                charge,
                exp_mz,
            )
            exp_idx = cand_idx + 1
        elif left_idx > exp_idx:
            exp_idx = left_idx

    return matched_peaks_count, matched_peaks


def _match_group_candidates(
    theoretical_peaks: list[TheoreticalIonWithMass],
    theo_idx: int,
    exp_mz: float,
    tolerance: Tolerance,
) -> tuple[int, list[TheoreticalIonWithMass]]:
    """Collect all consecutive theoretical peaks matching the same experimental peak.

    Returns:
        Tuple of (new_theo_idx, candidates).
    """
    candidates = []
    while theo_idx < len(theoretical_peaks):
        cand = theoretical_peaks[theo_idx]
        if not tolerance.is_within(cand.mz, exp_mz):
            break
        candidates.append(cand)
        theo_idx += 1
    return theo_idx, candidates


def _score_missing_cluster(
    theoretical_peaks: list[TheoreticalIonWithMass],
    theo_idx: int,
    exp_mzs: tuple[float, ...],
    exp_idx: int,
    tolerance: Tolerance,
) -> tuple[int, float]:
    """Score a cluster of missing theoretical peaks starting at theo_idx.

    Returns:
        Tuple of (new_theo_idx, score_contribution).
    """
    cluster_start = theoretical_peaks[theo_idx]
    missing_candidates = [cluster_start]
    theo_idx += 1

    while theo_idx < len(theoretical_peaks):
        cand = theoretical_peaks[theo_idx]
        tol = tolerance.get_limit(cand.mz)

        if abs(cand.mz - cluster_start.mz) >= 2 * tol:
            break

        if exp_idx < len(exp_mzs):
            current_mz = exp_mzs[exp_idx]
            if tolerance.is_within(cand.mz, current_mz):
                break

        missing_candidates.append(cand)
        theo_idx += 1

    rep = min(missing_candidates, key=lambda c: c.charge)
    charge_weights = SCORING_WEIGHTS[rep.charge]
    return theo_idx, charge_weights.missing_peak_weight


def _match_peaks_intensity(
    spec_index: ExperimentalSpectrumIndex,
    theory_ions_with_mass: list[TheoreticalIonWithMass],
    max_charge: int,
    tolerance: Tolerance,
) -> tuple[float, list[MatchedPeak]]:
    """Peak matching algorithm with intensity-based scoring using the spectrum index.

    Args:
        spec_index: Pre-indexed experimental spectrum.
        theory_ions_with_mass: Sorted list of theoretical ions with pre-calculated masses.
        max_charge: Maximum charge to consider.
        tolerance: Tolerance configuration.

    Returns:
        Tuple of (score, matched_peaks).
    """
    theoretical_peaks = [ion for ion in theory_ions_with_mass if ion.charge in SCORING_WEIGHTS]
    exp_mzs = spec_index.mzs_list
    exp_ranks = spec_index.ranks
    n_exp = len(exp_mzs)

    score = 0.0
    matched_peaks: list[MatchedPeak] = []

    exp_idx = 0
    theo_idx = 0

    while theo_idx < len(theoretical_peaks):
        ion = theoretical_peaks[theo_idx]
        mz = ion.mz
        tol = tolerance.get_limit(mz)

        left_idx = bisect.bisect_left(exp_mzs, mz - tol, lo=exp_idx)
        left_idx = get_left_index(spec_index, mz - tol)
        if exp_idx < left_idx:
            exp_idx = left_idx

        if exp_idx < n_exp and exp_mzs[exp_idx] <= mz + tol:
            exp_mz = exp_mzs[exp_idx]
            theo_idx, candidates = _match_group_candidates(
                theoretical_peaks, theo_idx, exp_mz, tolerance
            )
            rep = min(candidates, key=lambda c: c.charge)

            charge_weights = SCORING_WEIGHTS[rep.charge]
            rank = min(int(exp_ranks[exp_idx]), charge_weights.max_rank)
            score += charge_weights.weights[rank]

            matched_peaks.append(
                MatchedPeak(
                    theoretical_mz=rep.mz,
                    experimental_mz=exp_mz,
                    experimental_peak_intensity=float(spec_index.intensities[exp_idx]),
                    experimental_peak_idx=exp_idx,
                    charge=rep.charge,
                    theoretical_fragment_mask=rep.mask,
                )
            )
            logger.debug(
                "Matched ion '%s' (m/z: %f, charge: %d) as rep to peak (m/z: %f)",
                rep.fragment_id,
                rep.mz,
                rep.charge,
                exp_mz,
            )
            exp_idx += 1
            continue

        theo_idx, missing_score = _score_missing_cluster(
            theoretical_peaks, theo_idx, exp_mzs, exp_idx, tolerance
        )
        score += missing_score

    return score, matched_peaks


def _get_matched_peaks_count(
    scoring: ScoringMode,
    matched_peaks: list[MatchedPeak],
    matched_peaks_count: int,
) -> int:
    """Calculate match score based on the selected scoring mode."""
    return matched_peaks_count
