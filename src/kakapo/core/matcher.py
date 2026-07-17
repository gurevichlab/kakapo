"""Spectrum scoring engine for matching experimental against theoretical data."""

import bisect
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace

from tqdm import tqdm

from kakapo.config import DEFAULT_MIN_PEAKS, MAX_TOTAL_MASS_DIFFERENCE
from kakapo.constants import MAX_CHARGE, PROTON_MASS, SCORING_WEIGHTS
from kakapo.models import (
    MatchedPeak,
    MatchResult,
    Peak,
    ScoringMode,
    Spectrum,
    TheoreticalIonWithMass,
    TheoreticalSpectrum,
    Tolerance,
)
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def _fast_dot_product(vector: Iterable[float], mask: int) -> float:
    """Fast dot product of a float vector and a bitmask.

    Args:
        vector: An iterable of floats (e.g. monomer masses).
        mask: An integer bitmask representing which monomers are present.

    Returns:
        The sum of the masses of the active monomers.
    """
    total = 0.0
    for val in vector:
        if not mask:
            break
        if mask & 1:
            total += val
        mask >>= 1
    return total


def _match_peaks(
    exp_peaks: list[Peak],
    theory_ions_with_mass: list[TheoreticalIonWithMass],
    max_charge: int,
    tolerance: Tolerance,
) -> tuple[int, list[MatchedPeak]]:
    """Greedy peak matching algorithm of pre-calculated theoretical ions against experimental
    peaks.

    Args:
        exp_peaks: Sorted list of experimental peaks.
        theory_ions_with_mass: Sorted list of theoretical ions with pre-calculated masses.
        max_charge: Maximum charge to consider.
        tolerance: Tolerance configuration.

    Returns:
        Tuple of (matched_peaks_count, matched_peaks).
    """
    matched_peaks_count = 0
    matched_peaks: list[MatchedPeak] = []

    # Two pointers technique to find matches
    exp_idx_map = dict.fromkeys(range(1, max_charge + 1), 0)
    theo_idx = 0

    while theo_idx < len(theory_ions_with_mass):
        ion = theory_ions_with_mass[theo_idx]
        mz = ion.mz
        charge = ion.charge

        if charge not in exp_idx_map:
            theo_idx += 1
            continue

        exp_idx = exp_idx_map[charge]
        tol = tolerance.get_limit(mz)

        # Advance experimental index past peaks below the current tolerance window
        while exp_idx < len(exp_peaks):
            peak = exp_peaks[exp_idx]
            if peak.mass < mz - tol:
                exp_idx += 1
            else:
                break
        exp_idx_map[charge] = exp_idx

        # If we have an experimental peak in range, check for a match
        if exp_idx < len(exp_peaks):
            peak = exp_peaks[exp_idx]
            if tolerance.is_within(mz, peak.mass):
                matched_peaks_count += 1
                matched_peaks.append(
                    MatchedPeak(
                        theoretical_mz=mz,
                        experimental_mz=peak.mass,
                        experimental_peak_idx=exp_idx,
                        charge=charge,
                        theoretical_fragment_mask=ion.mask,
                    )
                )
                logger.debug(
                    "Matched ion '%s' (m/z: %f, charge: %d) to peak (m/z: %f)",
                    ion.fragment_id,
                    mz,
                    charge,
                    peak.mass,
                )
                exp_idx_map[charge] = exp_idx + 1

        theo_idx += 1

    return matched_peaks_count, matched_peaks


def _match_group_candidates(
    theoretical_peaks: list[TheoreticalIonWithMass],
    theo_idx: int,
    peak: Peak,
    tolerance: Tolerance,
) -> tuple[int, list[TheoreticalIonWithMass]]:
    """Collect all consecutive theoretical peaks matching the same experimental peak.

    Returns:
        Tuple of (new_theo_idx, candidates).
    """
    candidates = []
    while theo_idx < len(theoretical_peaks):
        cand = theoretical_peaks[theo_idx]
        if not tolerance.is_within(cand.mz, peak.mass):
            break
        candidates.append(cand)
        theo_idx += 1
    return theo_idx, candidates


def _score_missing_cluster(
    theoretical_peaks: list[TheoreticalIonWithMass],
    theo_idx: int,
    exp_peaks: list[Peak],
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

        if exp_idx < len(exp_peaks):
            current_peak = exp_peaks[exp_idx]
            if tolerance.is_within(cand.mz, current_peak.mass):
                break

        missing_candidates.append(cand)
        theo_idx += 1

    rep = min(missing_candidates, key=lambda c: c.charge)
    charge_weights = SCORING_WEIGHTS[rep.charge]
    return theo_idx, charge_weights.missing_peak_weight


def _match_peaks_intencity(
    exp_peaks: list[Peak],
    theory_ions_with_mass: list[TheoreticalIonWithMass],
    max_charge: int,
    tolerance: Tolerance,
) -> tuple[float, list[MatchedPeak]]:
    """Two-pointer peak matching algorithm with intensity-based scoring.

    Args:
        exp_peaks: Sorted list of experimental peaks.
        theory_ions_with_mass: Sorted list of theoretical ions with pre-calculated masses.
        max_charge: Maximum charge to consider.
        tolerance: Tolerance configuration.

    Returns:
        Tuple of (score, matched_peaks).
    """
    # Filter candidate theoretical peaks for those with valid scoring weights
    theoretical_peaks = [ion for ion in theory_ions_with_mass if ion.charge in SCORING_WEIGHTS]

    # Map experimental peaks to intensity rank (descending intensity)
    sorted_by_intensity = sorted(exp_peaks, key=lambda p: p.intensity, reverse=True)
    rank_map = {peak: rank for rank, peak in enumerate(sorted_by_intensity)}

    score = 0.0
    matched_peaks: list[MatchedPeak] = []

    exp_idx = 0
    theo_idx = 0

    while theo_idx < len(theoretical_peaks):
        ion = theoretical_peaks[theo_idx]
        mz = ion.mz

        # Advance experimental index past peaks below the current tolerance window
        while exp_idx < len(exp_peaks):
            peak = exp_peaks[exp_idx]
            tol = tolerance.get_limit(mz)
            if peak.mass < mz - tol:
                exp_idx += 1
            else:
                break

        # If we have an experimental peak in range, check for a match
        if exp_idx < len(exp_peaks):
            peak = exp_peaks[exp_idx]
            if tolerance.is_within(mz, peak.mass):
                theo_idx, candidates = _match_group_candidates(
                    theoretical_peaks, theo_idx, peak, tolerance
                )
                rep = min(candidates, key=lambda c: c.charge)

                charge_weights = SCORING_WEIGHTS[rep.charge]
                rank = min(rank_map[peak], charge_weights.max_rank)
                score += charge_weights.weights[rank]

                matched_peaks.append(
                    MatchedPeak(
                        theoretical_mz=rep.mz,
                        experimental_mz=peak.mass,
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
                    peak.mass,
                )
                exp_idx += 1
                continue

        # The current theoretical peak did not match. Find missing candidates group.
        theo_idx, missing_score = _score_missing_cluster(
            theoretical_peaks, theo_idx, exp_peaks, exp_idx, tolerance
        )
        score += missing_score

    return score, matched_peaks


def _calculate_theory_ions_with_mass(
    theoretical_spectrum: TheoreticalSpectrum,
    monomer_masses: Iterable[float],
    max_charge: int,
) -> list[TheoreticalIonWithMass]:
    """Calculate and sort theoretical ion masses based on monomer masses and charges."""
    ions = []
    for charge in range(1, max_charge + 1):
        for ion in theoretical_spectrum.ions:
            mass = _fast_dot_product(monomer_masses, ion.mask) + ion.total_shift_mass
            mz = (mass + PROTON_MASS * charge) / charge
            ions.append(
                TheoreticalIonWithMass(
                    fragment_id=ion.fragment_id,
                    mass=mass,
                    mask=ion.mask,
                    charge=charge,
                    mz=mz,
                )
            )
    return sorted(ions, key=lambda item: item.mz)


def _calculate_score(
    scoring: ScoringMode,
    matched_peaks: list[MatchedPeak],
    matched_peaks_count: int,
) -> int:
    """Calculate match score based on the selected scoring mode."""
    if scoring == ScoringMode.EXP_PEAKS:
        return len({mp.experimental_mz for mp in matched_peaks})
    return matched_peaks_count


def _match_spectrum_unmodified(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    tolerance: Tolerance,
    mass_mod: float,
    scoring: ScoringMode,
) -> MatchResult:
    """Helper to match spectrum using unmodified monomer masses."""
    # Sort to ensure reproducible greedy matching
    exp_peaks = sorted(spectrum.peaks, key=lambda p: p.mass)
    monomer_masses = theoretical_spectrum.monomer_mass_vector
    max_charge = spectrum.charge if spectrum.charge else MAX_CHARGE

    logger.debug(
        "Spectrum '%s' peaks: %d. Theoretical spectrum '%s' ions: %d. Max charge: %d. "
        "Precursor: %f, Structure: %f, mass_mod: %f",
        spectrum.spectrum_id,
        len(exp_peaks),
        theoretical_spectrum.structure_id,
        len(theoretical_spectrum.ions),
        max_charge,
        spectrum.precursor_mass,
        theoretical_spectrum.structure_mass,
        mass_mod,
    )

    theory_ions_with_mass = _calculate_theory_ions_with_mass(
        theoretical_spectrum, monomer_masses, max_charge
    )

    if scoring == ScoringMode.INTENCITY:
        score, matched_peaks = _match_peaks_intencity(
            exp_peaks, theory_ions_with_mass, max_charge, tolerance
        )
        matched_peaks_count = len(matched_peaks)
    else:
        matched_peaks_count, matched_peaks = _match_peaks(
            exp_peaks, theory_ions_with_mass, max_charge, tolerance
        )
        score = _calculate_score(scoring, matched_peaks, matched_peaks_count)

    logger.debug(
        "Matching complete for '%s' vs '%s': matched %d peaks (unmodified).",
        spectrum.spectrum_id,
        theoretical_spectrum.structure_id,
        matched_peaks_count,
    )

    return MatchResult(
        structure_id=theoretical_spectrum.structure_id,
        spectrum_id=spectrum.spectrum_id,
        structure_mass=theoretical_spectrum.structure_mass,
        spectrum_mass=spectrum.precursor_mass,
        tolerance=tolerance,
        matched_peaks_count=matched_peaks_count,
        theoretical_peaks_count=len(theoretical_spectrum.ions),
        experimental_peaks_count=len(spectrum.peaks),
        monomer_count=theoretical_spectrum.monomer_count,
        unmodified_monomer_mass_vector=theoretical_spectrum.monomer_mass_vector,
        score=score,
        matched_peaks=matched_peaks,
        modified_monomer_idx=None,
        mass_mod=mass_mod,
        initial_modified_monomer_mass=None,
    )


def _match_spectrum_with_mod(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    tolerance: Tolerance,
    mass_mod: float,
    scoring: ScoringMode,
    x_mode: bool = False,
) -> MatchResult:
    """Helper to match spectrum using modification-tolerant monomer mass updates."""
    # Sort to ensure reproducible greedy matching
    exp_peaks = sorted(spectrum.peaks, key=lambda p: p.mass)
    monomer_masses = theoretical_spectrum.monomer_mass_vector
    max_charge = spectrum.charge if spectrum.charge else MAX_CHARGE

    logger.debug(
        "Spectrum '%s' peaks: %d. Theoretical spectrum '%s' ions: %d. Max charge: %d. "
        "Precursor: %f, Structure: %f, mass_mod: %f",
        spectrum.spectrum_id,
        len(exp_peaks),
        theoretical_spectrum.structure_id,
        len(theoretical_spectrum.ions),
        max_charge,
        spectrum.precursor_mass,
        theoretical_spectrum.structure_mass,
        mass_mod,
    )

    max_score = float("-inf")
    max_matched_count = 0
    best_matched_peaks = []
    best_i = None

    for i in range(len(monomer_masses)):
        if x_mode:
            monomer = theoretical_spectrum.monomers.get(i)
            if monomer is None or not monomer.norine_name.startswith("X"):
                continue

        updated_monomer_masses = list(monomer_masses)
        updated_monomer_masses[i] += mass_mod
        if updated_monomer_masses[i] < 0:
            continue

        theory_ions_with_mass = _calculate_theory_ions_with_mass(
            theoretical_spectrum, updated_monomer_masses, max_charge
        )

        if scoring == ScoringMode.INTENCITY:
            score, matched_peaks = _match_peaks_intencity(
                exp_peaks, theory_ions_with_mass, max_charge, tolerance
            )
            matched_peaks_count = len(matched_peaks)
        else:
            matched_peaks_count, matched_peaks = _match_peaks(
                exp_peaks, theory_ions_with_mass, max_charge, tolerance
            )
            score = float(_calculate_score(scoring, matched_peaks, matched_peaks_count))

        if matched_peaks_count > 0:
            if best_i is None or score > max_score:
                max_score = score
                max_matched_count = matched_peaks_count
                best_matched_peaks = matched_peaks
                best_i = i

    logger.debug(
        "Matching complete for '%s' vs '%s': max matched %d peaks (best monomer index: %s).",
        spectrum.spectrum_id,
        theoretical_spectrum.structure_id,
        max_matched_count,
        str(best_i),
    )

    initial_modified_monomer_mass = monomer_masses[best_i] if best_i is not None else None
    final_score = max_score if best_i is not None else 0.0

    return MatchResult(
        structure_id=theoretical_spectrum.structure_id,
        spectrum_id=spectrum.spectrum_id,
        structure_mass=theoretical_spectrum.structure_mass,
        spectrum_mass=spectrum.precursor_mass,
        tolerance=tolerance,
        matched_peaks_count=max_matched_count,
        theoretical_peaks_count=len(theoretical_spectrum.ions),
        experimental_peaks_count=len(spectrum.peaks),
        monomer_count=theoretical_spectrum.monomer_count,
        unmodified_monomer_mass_vector=monomer_masses,
        matched_peaks=best_matched_peaks,
        modified_monomer_idx=best_i,
        mass_mod=mass_mod,
        initial_modified_monomer_mass=initial_modified_monomer_mass,
        score=final_score,
    )


def match_spectrum(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    tolerance: Tolerance,
    scoring: ScoringMode,
    x_mode: bool = False,
) -> MatchResult:
    """Match an experimental spectrum to a theoretical spectrum.

    If the precursor mass of the spectrum matches the structure mass within tolerance,
    it runs the unmodified matching algorithm without searching for modifications.
    Otherwise, it runs the modification-tolerant match algorithm, searching for
    the best monomer to modify by mass_mod = precursor - structure mass.

    Args:
        spectrum: The experimental Spectrum instance.
        theoretical_spectrum: The TheoreticalSpectrum instance.
        tolerance: The Tolerance configuration allowed for a match.

    Returns:
        A MatchResult instance containing matching metrics.
    """
    logger.debug(
        "Matching experimental spectrum '%s' against structure '%s' (tolerance: %s)",
        spectrum.spectrum_id,
        theoretical_spectrum.structure_id,
        tolerance,
    )

    precursor_neutral_mass = spectrum.get_neutral_mass()
    mass_mod = precursor_neutral_mass - theoretical_spectrum.structure_mass

    precursor_mz = spectrum.precursor_mass
    precursor_tol = tolerance.get_limit(precursor_mz)
    charge = spectrum.charge if spectrum.charge is not None else 1
    precursor_neutral_tolerance = charge * precursor_tol

    if abs(mass_mod) <= precursor_neutral_tolerance:
        logger.debug(
            "Precursor neutral mass (%f) matches structure mass (%f) within tolerance (%f). "
            "Running unmodified matching.",
            precursor_neutral_mass,
            theoretical_spectrum.structure_mass,
            precursor_neutral_tolerance,
        )
        return _match_spectrum_unmodified(
            spectrum, theoretical_spectrum, tolerance, mass_mod, scoring
        )

    return _match_spectrum_with_mod(
        spectrum, theoretical_spectrum, tolerance, mass_mod, scoring, x_mode=x_mode
    )


@dataclass
class _StructureIndex:
    """Index structure for fast theoretical spectrum peak lookup.

    Attributes:
        bin_width: Bin width in Daltons.
        sorted_peaks: List of sorted tuples (theoretical_ion_mass, structure_id).
        index_table: Mapping from bin index to offset in sorted_peaks.
    """

    bin_width: float
    sorted_peaks: list[tuple[float, str]]
    index_table: list[int]


def _build_structure_index(
    theoretical_spectra: Iterable[TheoreticalSpectrum],
    bin_width: float = 0.2,
) -> _StructureIndex:
    """Build a search index over theoretical spectra peaks.

    Args:
        theoretical_spectra: Theoretical spectrum objects to index.
        bin_width: Bin width in Daltons (defaults to 0.2 Da).

    Returns:
        A _StructureIndex container holding the index state.
    """
    flat_peaks: list[tuple[float, str]] = []
    for theo in theoretical_spectra:
        ions = _calculate_theory_ions_with_mass(theo, theo.monomer_mass_vector, 1)
        for ion in ions:
            flat_peaks.append((ion.mass, theo.structure_id))

    flat_peaks.sort(key=lambda item: item[0])

    if not flat_peaks:
        return _StructureIndex(bin_width=bin_width, sorted_peaks=[], index_table=[0])

    max_mass = flat_peaks[-1][0]
    num_bins = int(max_mass // bin_width) + 3

    index_table: list[int] = []
    for i in range(num_bins):
        target = i * bin_width
        idx = bisect.bisect_left(flat_peaks, (target, ""))
        index_table.append(idx)

    return _StructureIndex(bin_width=bin_width, sorted_peaks=flat_peaks, index_table=index_table)


def _filter_candidates_with_index(
    spectrum: Spectrum,
    index: _StructureIndex,
    tolerance: Tolerance,
    min_shared_peaks: int = DEFAULT_MIN_PEAKS,
) -> set[str]:
    """Query experimental spectrum against the index to filter candidate structures.

    The filtering algorithm proceeds as follows:
    1. For each experimental peak, iterate through all possible precursor charge states `c` from 1
       to `max_charge` to compute the candidate neutral theoretical mass and mass tolerance.
    2. Query the search index for two potential candidate neutral masses:
       - The unmodified neutral mass (for unmodified fragments)
       - The complement neutral mass (for modified fragments, by matching their unmodified
         complements)
    3. Aggregate hits per structure in a shared peak counter, and return the set of structure IDs
       with total shared peak counts >= `min_shared_peaks`.

    Args:
        spectrum: The experimental Spectrum instance.
        index: The _StructureIndex compile of database spectra.
        tolerance: Tolerance configuration.
        min_shared_peaks: Minimum Shared Peak Count score to keep a candidate
            structure (defaults to DEFAULT_MIN_PEAKS).

    Returns:
        A set of structure_ids matching the threshold.
    """
    if not spectrum.peaks or not index.sorted_peaks:
        return set()

    max_charge = spectrum.charge if spectrum.charge is not None else MAX_CHARGE
    hits: Counter[str] = Counter()
    precursor_neutral_mass = spectrum.get_neutral_mass()

    def query_index(target_mass: float, tolerance: float) -> None:
        """Query the search index for a target mass and increment hits for matching structures.

        Uses binary search over the indexed theoretical peaks within a bounded slice to find all
        structures which theoretical peaks fall within the mass tolerance window.

        Args:
            target_mass: The candidate neutral mass to search for.
            tolerance: The absolute mass tolerance window in Daltons.
        """
        # Calculate the lower bin boundary using a direct floor division
        i_lower = int((target_mass - tolerance) // index.bin_width)
        if i_lower < 0:
            i_lower = 0
        if i_lower >= len(index.index_table):
            return

        # Calculate the upper bin boundary to dynamically size the search slice
        i_upper = int((target_mass + tolerance) // index.bin_width)

        # Bounded Binary Search Slice
        start_idx = index.index_table[i_lower]
        end_idx = index.index_table[min(i_upper + 1, len(index.index_table) - 1)]

        left_idx = bisect.bisect_left(
            index.sorted_peaks,
            (target_mass - tolerance, ""),
            lo=start_idx,
            hi=end_idx,
        )
        right_idx = bisect.bisect_right(
            index.sorted_peaks,
            (target_mass + tolerance, "z" * 100),
            lo=start_idx,
            hi=end_idx,
        )

        for idx in range(left_idx, right_idx):
            _, struct_id = index.sorted_peaks[idx]
            hits[struct_id] += 1

    for peak in spectrum.peaks:
        for c in range(1, max_charge + 1):
            # Convert experimental peak m/z to neutral mass at charge c
            neutral_mass = peak.mass * c - PROTON_MASS * c
            tol_mz = tolerance.get_limit(peak.mass)
            neutral_tolerance = c * tol_mz

            # 1. Unmodified peak match
            query_index(neutral_mass, neutral_tolerance)

            # 2. Modified peak match (via its unmodified complement fragment)
            complement_mass = precursor_neutral_mass - neutral_mass
            if complement_mass > neutral_tolerance:
                query_index(complement_mass, neutral_tolerance)

    return {struct_id for struct_id, count in hits.items() if count >= min_shared_peaks}


def match_spectra(
    spectra: Iterable[Spectrum],
    theoretical_spectra: Iterable[TheoreticalSpectrum],
    tolerance: Tolerance,
    scoring: ScoringMode,
    min_shared_peaks: int = DEFAULT_MIN_PEAKS,
    x_mode: bool = False,
) -> list[MatchResult]:
    """Match each experimental spectrum to each theoretical spectrum pairwise.

    Args:
        spectra: An iterable of experimental Spectrum instances.
        theoretical_spectra: An iterable of TheoreticalSpectrum instances.
        tolerance: The Tolerance configuration.
        min_shared_peaks: Minimum shared peak count for candidate filtering
            (defaults to DEFAULT_MIN_PEAKS).

    Returns:
        A list of MatchResult instances for all pairwise matches.
    """
    logger.info("Starting batch matching (cross-matching each spectrum against each structure)...")
    spectra_list = list(spectra)
    theo_list = list(theoretical_spectra)
    logger.debug(
        "Loaded %d experimental spectra and %d theoretical spectra for matching",
        len(spectra_list),
        len(theo_list),
    )

    index = _build_structure_index(theo_list)
    theo_map = {theo.structure_id: theo for theo in theo_list}

    results: list[MatchResult] = []
    for spec in tqdm(spectra_list, desc="Matching spectra", unit="spec"):
        feasible_ids = _filter_candidates_with_index(
            spec, index, tolerance, min_shared_peaks=min_shared_peaks
        )
        if min_shared_peaks <= 1:
            for struct_id, theo in theo_map.items():
                if theo.monomer_count == 1:
                    charges_to_try = (
                        [spec.charge] if spec.charge is not None else list(range(1, MAX_CHARGE + 1))
                    )
                    if any(
                        abs(theo.structure_mass - (spec.precursor_mass * z - PROTON_MASS * z))
                        <= MAX_TOTAL_MASS_DIFFERENCE
                        for z in charges_to_try
                    ):
                        feasible_ids.add(struct_id)

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
                        spec_with_charge, theo, tolerance, scoring, x_mode=x_mode
                    )
                    if result.matched_peaks_count > 0:
                        results.append(result)

    logger.info(
        "Completed batch matching. Generated %d total MatchResult(s).",
        len(results),
    )
    return results
