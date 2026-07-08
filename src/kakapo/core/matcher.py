"""Spectrum scoring engine for matching experimental against theoretical data."""

import bisect
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from kakapo.config import DEFAULT_MIN_PEAKS, MAX_TOTAL_MASS_DIFFERENCE
from kakapo.constants import MAX_CHARGE, PROTON_MASS
from kakapo.models import MatchedPeak, MatchResult, Peak, Spectrum, TheoreticalSpectrum
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
    theory_ions_with_mass: list[tuple[str, float]],
    max_charge: int,
    mass_tolerance: float,
) -> tuple[int, list[MatchedPeak]]:
    """Greedy peak matching algorithm of pre-calculated theoretical ions against experimental
    peaks.

    Args:
        exp_peaks: Sorted list of experimental peaks.
        theory_ions_with_mass: Sorted list of theoretical ions with pre-calculated masses.
        max_charge: Maximum charge to consider.
        mass_tolerance: Maximum absolute mass difference allowed for a match.

    Returns:
        Tuple of (matched_peaks_count, matched_peaks).
    """
    matched_peaks_count = 0
    matched_peaks: list[MatchedPeak] = []

    for charge in range(1, max_charge + 1):
        # Two pointers technique to find matches
        last_idx = 0
        for frag_id, ion_mass in theory_ions_with_mass:
            mz = (ion_mass + PROTON_MASS * charge) / charge
            for idx in range(last_idx, len(exp_peaks)):
                peak = exp_peaks[idx]
                if abs(peak.mass - mz) <= mass_tolerance:
                    matched_peaks_count += 1
                    last_idx = idx + 1
                    matched_peaks.append(
                        MatchedPeak(
                            theoretical_mz=mz,
                            experimental_mz=peak.mass,
                            charge=charge,
                        )
                    )
                    logger.debug(
                        "Matched ion '%s' (m/z: %f, charge: %d) to peak (m/z: %f)",
                        frag_id,
                        mz,
                        charge,
                        peak.mass,
                    )
                    break
                if peak.mass > mz + mass_tolerance:
                    last_idx = idx
                    break
            else:
                last_idx += 1
    return matched_peaks_count, matched_peaks


def _calculate_theory_ions_with_mass(
    theoretical_spectrum: TheoreticalSpectrum,
    monomer_masses: Iterable[float],
) -> list[tuple[str, float]]:
    """Calculate and sort theoretical ion masses based on monomer masses."""
    return sorted(
        [
            (
                ion.fragment_id,
                _fast_dot_product(monomer_masses, ion.mask) + ion.total_shift_mass,
            )
            for ion in theoretical_spectrum.ions
        ],
        key=lambda item: item[1],
    )


def _match_spectrum_unmodified(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    mass_tolerance: float,
    mass_mod: float,
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

    theory_ions_with_mass = _calculate_theory_ions_with_mass(theoretical_spectrum, monomer_masses)

    matched_peaks_count, matched_peaks = _match_peaks(
        exp_peaks, theory_ions_with_mass, max_charge, mass_tolerance
    )

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
        mass_tolerance=mass_tolerance,
        matched_peaks_count=matched_peaks_count,
        theoretical_peaks_count=len(theoretical_spectrum.ions),
        experimental_peaks_count=len(spectrum.peaks),
        monomer_count=theoretical_spectrum.monomer_count,
        matched_peaks=matched_peaks,
        modified_monomer_idx=None,
        mass_mod=mass_mod,
        initial_modified_monomer_mass=None,
    )


def _match_spectrum_with_mod(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    mass_tolerance: float,
    mass_mod: float,
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

    max_matched_count = 0
    best_matched_peaks = []
    best_i = None

    for i in range(len(monomer_masses)):
        updated_monomer_masses = list(monomer_masses)
        updated_monomer_masses[i] += mass_mod
        if updated_monomer_masses[i] <= 0:
            continue

        theory_ions_with_mass = _calculate_theory_ions_with_mass(
            theoretical_spectrum, updated_monomer_masses
        )

        matched_peaks_count, matched_peaks = _match_peaks(
            exp_peaks, theory_ions_with_mass, max_charge, mass_tolerance
        )

        if matched_peaks_count > max_matched_count:
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

    return MatchResult(
        structure_id=theoretical_spectrum.structure_id,
        spectrum_id=spectrum.spectrum_id,
        structure_mass=theoretical_spectrum.structure_mass,
        spectrum_mass=spectrum.precursor_mass,
        mass_tolerance=mass_tolerance,
        matched_peaks_count=max_matched_count,
        theoretical_peaks_count=len(theoretical_spectrum.ions),
        experimental_peaks_count=len(spectrum.peaks),
        monomer_count=theoretical_spectrum.monomer_count,
        matched_peaks=best_matched_peaks,
        modified_monomer_idx=best_i,
        mass_mod=mass_mod,
        initial_modified_monomer_mass=initial_modified_monomer_mass,
    )


def match_spectrum(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    mass_tolerance: float,
) -> MatchResult:
    """Match an experimental spectrum to a theoretical spectrum.

    If the precursor mass of the spectrum matches the structure mass within tolerance,
    it runs the unmodified matching algorithm without searching for modifications.
    Otherwise, it runs the modification-tolerant match algorithm, searching for
    the best monomer to modify by mass_mod = precursor - structure mass.

    Args:
        spectrum: The experimental Spectrum instance.
        theoretical_spectrum: The TheoreticalSpectrum instance.
        mass_tolerance: The maximum absolute mass difference (in Da) allowed for a match.

    Returns:
        A MatchResult instance containing matching metrics.
    """
    logger.debug(
        "Matching experimental spectrum '%s' against structure '%s' (tolerance: %f Da)",
        spectrum.spectrum_id,
        theoretical_spectrum.structure_id,
        mass_tolerance,
    )

    precursor_neutral_mass = spectrum.get_neutral_mass()
    mass_mod = precursor_neutral_mass - theoretical_spectrum.structure_mass

    if abs(mass_mod) <= mass_tolerance:
        logger.debug(
            "Precursor neutral mass (%f) matches structure mass (%f) within tolerance (%f). "
            "Running unmodified matching.",
            precursor_neutral_mass,
            theoretical_spectrum.structure_mass,
            mass_tolerance,
        )
        return _match_spectrum_unmodified(spectrum, theoretical_spectrum, mass_tolerance, mass_mod)

    return _match_spectrum_with_mod(spectrum, theoretical_spectrum, mass_tolerance, mass_mod)


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
        ions = _calculate_theory_ions_with_mass(theo, theo.monomer_mass_vector)
        for _, mass in ions:
            flat_peaks.append((mass, theo.structure_id))

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
    mass_tolerance: float,
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
        mass_tolerance: Mass tolerance in Daltons for peaks matching.
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
            neutral_tolerance = c * mass_tolerance

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
    mass_tolerance: float,
    min_shared_peaks: int = DEFAULT_MIN_PEAKS,
) -> list[MatchResult]:
    """Match each experimental spectrum to each theoretical spectrum pairwise.

    Args:
        spectra: An iterable of experimental Spectrum instances.
        theoretical_spectra: An iterable of TheoreticalSpectrum instances.
        mass_tolerance: The maximum absolute mass difference (in Da) allowed for a match.
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
    for spec in spectra_list:
        feasible_ids = _filter_candidates_with_index(
            spec, index, mass_tolerance, min_shared_peaks=min_shared_peaks
        )
        if min_shared_peaks <= 1:
            for struct_id, theo in theo_map.items():
                if theo.monomer_count == 1:
                    if (
                        abs(theo.structure_mass - spec.get_neutral_mass())
                        <= MAX_TOTAL_MASS_DIFFERENCE
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
            if abs(theo.structure_mass - spec.get_neutral_mass()) <= MAX_TOTAL_MASS_DIFFERENCE:
                result = match_spectrum(spec, theo, mass_tolerance)
                if result.matched_peaks_count > 0:
                    results.append(result)

    logger.info(
        "Completed batch matching. Generated %d total MatchResult(s).",
        len(results),
    )
    return results
