"""Spectrum scoring engine for matching experimental against theoretical data."""

import bisect
import itertools
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

from tqdm import tqdm

from kakapo.config import DEFAULT_MIN_PEAKS, MAX_TOTAL_MASS_DIFFERENCE
from kakapo.constants import MAX_CHARGE, PROTON_MASS, RESIDUE_MODIFICATIONS, SCORING_WEIGHTS
from kakapo.models import (
    MatchedPeak,
    MatchResult,
    ModificationHypothesis,
    ModificationMode,
    MonomerModification,
    Peak,
    ScoringMode,
    Spectrum,
    TheoreticalIonWithMass,
    TheoreticalSpectrum,
    Tolerance,
)
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentalSpectrumIndex:
    """Search index for an experimental spectrum to enable fast peak matching.

    Attributes:
        sorted_peaks: Tuple of Peak objects sorted ascending by m/z mass.
        masses: Tuple of float peak m/z masses corresponding to sorted_peaks.
        ranks: Tuple of integer intensity ranks (0 = highest intensity).
        bin_width: Width of mass bins in Daltons for O(1) table lookup.
        index_table: Tuple of start indices in `sorted_peaks` for each bin.
    """

    sorted_peaks: tuple[Peak, ...]
    masses: tuple[float, ...]
    ranks: tuple[int, ...]
    bin_width: float
    index_table: tuple[int, ...]


def build_experimental_spectrum_index(
    spectrum: Spectrum, bin_width: float = 0.5
) -> ExperimentalSpectrumIndex:
    """Build a fast search index over an experimental spectrum's peaks.

    Args:
        spectrum: The experimental Spectrum instance.
        bin_width: Bin width in Daltons (defaults to 0.5 Da).

    Returns:
        An ExperimentalSpectrumIndex container holding the index state.
    """
    sorted_peaks = tuple(sorted(spectrum.peaks, key=lambda p: p.mass))
    masses = tuple(p.mass for p in sorted_peaks)

    # Precalculate intensity ranks (0 = most intense peak)
    sorted_by_intensity = sorted(
        enumerate(sorted_peaks), key=lambda item: item[1].intensity, reverse=True
    )
    rank_arr = [0] * len(sorted_peaks)
    for rank, (orig_idx, _) in enumerate(sorted_by_intensity):
        rank_arr[orig_idx] = rank
    ranks = tuple(rank_arr)

    if not sorted_peaks:
        return ExperimentalSpectrumIndex(
            sorted_peaks=(),
            masses=(),
            ranks=(),
            bin_width=bin_width,
            index_table=(0,),
        )

    max_mass = masses[-1]
    num_bins = int(max_mass // bin_width) + 3

    index_table: list[int] = []
    for i in range(num_bins):
        target = i * bin_width
        idx = bisect.bisect_left(masses, target)
        index_table.append(idx)

    return ExperimentalSpectrumIndex(
        sorted_peaks=sorted_peaks,
        masses=masses,
        ranks=ranks,
        bin_width=bin_width,
        index_table=tuple(index_table),
    )


def get_left_index(index: ExperimentalSpectrumIndex, min_mass: float) -> int:
    """Find the lower bound index in index.sorted_peaks for min_mass.

    Args:
        index: The experimental spectrum index.
        min_mass: Minimum mass threshold.

    Returns:
        The first index where peak mass >= min_mass.
    """
    if not index.masses or min_mass > index.masses[-1]:
        return len(index.masses)
    if min_mass <= index.masses[0]:
        return 0

    i_lower = int(min_mass // index.bin_width)
    if i_lower >= len(index.index_table):
        return len(index.masses)
    if i_lower < 0:
        i_lower = 0

    start_search = index.index_table[i_lower]
    return bisect.bisect_left(index.masses, min_mass, lo=start_search)


def get_matching_range(
    index: ExperimentalSpectrumIndex, mz: float, tolerance: Tolerance
) -> tuple[int, int]:
    """Find index bounds [start_idx, end_idx) of experimental peaks matching theoretical m/z.

    Args:
        index: The experimental spectrum index.
        mz: Theoretical m/z value.
        tolerance: Tolerance configuration (da or ppm).

    Returns:
        A tuple of (start_idx, end_idx) in index.sorted_peaks.
    """
    if not index.masses:
        return 0, 0

    tol = tolerance.get_limit(mz)
    min_mass = mz - tol
    max_mass = mz + tol

    if max_mass < index.masses[0] or min_mass > index.masses[-1]:
        return 0, 0

    left_idx = get_left_index(index, min_mass)

    i_upper = int(max_mass // index.bin_width)
    end_search = index.index_table[min(i_upper + 1, len(index.index_table) - 1)]
    right_idx = bisect.bisect_right(index.masses, max_mass, lo=left_idx, hi=end_search)

    return left_idx, right_idx


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

    exp_idx_map = dict.fromkeys(range(1, max_charge + 1), 0)

    for ion in theory_ions_with_mass:
        mz = ion.mz
        charge = ion.charge

        if charge not in exp_idx_map:
            continue

        exp_idx = exp_idx_map[charge]
        tol = tolerance.get_limit(mz)
        min_mass = mz - tol
        max_mass = mz + tol

        left_idx = get_left_index(spec_index, min_mass)
        cand_idx = max(exp_idx, left_idx)

        if cand_idx < len(spec_index.masses) and spec_index.masses[cand_idx] <= max_mass:
            peak = spec_index.sorted_peaks[cand_idx]
            matched_peaks_count += 1
            matched_peaks.append(
                MatchedPeak(
                    theoretical_mz=mz,
                    experimental_mz=peak.mass,
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
                peak.mass,
            )
            exp_idx_map[charge] = cand_idx + 1
        elif left_idx > exp_idx:
            exp_idx_map[charge] = left_idx

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
    exp_peaks: tuple[Peak, ...],
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
    exp_peaks = spec_index.sorted_peaks
    exp_masses = spec_index.masses

    score = 0.0
    matched_peaks: list[MatchedPeak] = []

    exp_idx = 0
    theo_idx = 0

    while theo_idx < len(theoretical_peaks):
        ion = theoretical_peaks[theo_idx]
        mz = ion.mz
        tol = tolerance.get_limit(mz)

        left_idx = get_left_index(spec_index, mz - tol)
        if exp_idx < left_idx:
            exp_idx = left_idx

        if exp_idx < len(exp_masses) and exp_masses[exp_idx] <= mz + tol:
            peak = exp_peaks[exp_idx]
            theo_idx, candidates = _match_group_candidates(
                theoretical_peaks, theo_idx, peak, tolerance
            )
            rep = min(candidates, key=lambda c: c.charge)

            charge_weights = SCORING_WEIGHTS[rep.charge]
            rank = min(spec_index.ranks[exp_idx], charge_weights.max_rank)
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

        theo_idx, missing_score = _score_missing_cluster(
            theoretical_peaks, theo_idx, exp_peaks, exp_idx, tolerance
        )
        score += missing_score

    return score, matched_peaks


def _calculate_theory_ions_with_mass(
    theoretical_spectrum: TheoreticalSpectrum,
    max_charge: int,
    modifications: Iterable[MonomerModification] = (),
) -> list[TheoreticalIonWithMass]:
    """Calculate and sort theoretical ion masses based on monomer masses and charges.

    Uses precomputed unmodified masses to update mass in O(1) for modified monomers.
    """
    # Extract active modifications from the modifications parameter
    active_mods = [(mod.monomer_idx, mod.mass_difference) for mod in modifications]

    # Fast path: if no modifications and charge is <= MAX_CHARGE, return cached list
    if not active_mods and max_charge <= MAX_CHARGE:
        return list(theoretical_spectrum.unmodified_ions_with_mass_by_charge[max_charge])

    ions = []
    for charge in range(1, max_charge + 1):
        for ion in theoretical_spectrum.ions:
            mass = ion.unmodified_mass
            for monomer_idx, diff in active_mods:
                if ion.mask & (1 << monomer_idx):
                    mass += diff
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


def _generate_unmodified_hypotheses(
    theoretical_spectrum: TheoreticalSpectrum,
) -> Iterator[ModificationHypothesis]:
    """Generate the single hypothesis for unmodified spectrum matching."""
    yield ModificationHypothesis(
        updated_monomer_masses=list(theoretical_spectrum.monomer_mass_vector),
        modifications=(),
    )


def _generate_single_mod_hypotheses(
    theoretical_spectrum: TheoreticalSpectrum,
    mass_mod: float,
    x_mode_only: bool,
) -> Iterator[ModificationHypothesis]:
    """Generate candidate hypotheses for single-modification matching."""
    monomer_masses = theoretical_spectrum.monomer_mass_vector
    for i in range(len(monomer_masses)):
        monomer = theoretical_spectrum.monomers.get(i)
        norine_name = monomer.norine_name if monomer is not None else f"Mon_{i}"

        if x_mode_only and not norine_name.startswith("X"):
            continue

        updated_masses = list(monomer_masses)
        updated_masses[i] += mass_mod
        if updated_masses[i] < 0:
            continue

        mod = MonomerModification(
            monomer_idx=i,
            initial_monomer_name=norine_name,
            initial_monomer_mass=monomer_masses[i],
            mass_difference=mass_mod,
            new_monomer_name=None,
            formula_difference=None,
        )
        yield ModificationHypothesis(updated_monomer_masses=updated_masses, modifications=(mod,))


def _generate_two_mod_hypotheses(
    theoretical_spectrum: TheoreticalSpectrum,
    mass_mod: float,
) -> Iterator[ModificationHypothesis]:
    """Generate candidate hypotheses for two-modification matching (known mod + residual on X)."""

    monomer_masses = theoretical_spectrum.monomer_mass_vector
    monomers = theoretical_spectrum.monomers

    for j, monomer_j in monomers.items():
        norine_name = monomer_j.norine_name
        known_mods = RESIDUE_MODIFICATIONS.get(norine_name, [])
        if not known_mods:
            continue

        for mod_entry in known_mods:
            known_mod_mass = float(mod_entry["mass_difference"])
            residual_mass = mass_mod - known_mod_mass

            for k, monomer_k in monomers.items():
                if not monomer_k.norine_name.startswith("X"):
                    continue

                updated_masses = list(monomer_masses)
                updated_masses[j] += known_mod_mass
                updated_masses[k] += residual_mass

                if updated_masses[j] < 0 or updated_masses[k] < 0:
                    continue

                new_name_1 = (
                    mod_entry["modified_monomers"][0]
                    if mod_entry.get("modified_monomers")
                    else None
                )
                mod1 = MonomerModification(
                    monomer_idx=j,
                    initial_monomer_name=monomer_j.norine_name,
                    initial_monomer_mass=monomer_masses[j],
                    mass_difference=known_mod_mass,
                    new_monomer_name=new_name_1,
                    formula_difference=mod_entry.get("formula_difference"),
                )
                mod2 = MonomerModification(
                    monomer_idx=k,
                    initial_monomer_name=monomer_k.norine_name,
                    initial_monomer_mass=monomer_masses[k],
                    mass_difference=residual_mass,
                    new_monomer_name=None,
                    formula_difference=None,
                )
                yield ModificationHypothesis(
                    updated_monomer_masses=updated_masses, modifications=(mod1, mod2)
                )


def _evaluate_hypotheses(
    spec_index: ExperimentalSpectrumIndex,
    theoretical_spectrum: TheoreticalSpectrum,
    hypotheses: Iterable[ModificationHypothesis],
    tolerance: Tolerance,
    scoring: ScoringMode,
    spectrum_id: str,
    precursor_mass: float,
    charge: int | None,
) -> MatchResult:
    """Evaluate candidate modification hypotheses and return the best scoring MatchResult.

    Returns:
        A MatchResult instance for the optimal hypothesis.
    """
    max_charge = charge if charge is not None else MAX_CHARGE

    max_score = float("-inf")
    max_matched_count = 0
    best_matched_peaks: list[MatchedPeak] = []
    best_modifications: tuple[MonomerModification, ...] = ()
    found_any = False

    for hypothesis in hypotheses:
        theory_ions_with_mass = _calculate_theory_ions_with_mass(
            theoretical_spectrum, max_charge, hypothesis.modifications
        )

        if scoring == ScoringMode.INTENCITY:
            score, matched_peaks = _match_peaks_intencity(
                spec_index, theory_ions_with_mass, max_charge, tolerance
            )
            matched_peaks_count = len(matched_peaks)
        else:
            matched_peaks_count, matched_peaks = _match_peaks(
                spec_index, theory_ions_with_mass, max_charge, tolerance
            )
            score = float(_calculate_score(scoring, matched_peaks, matched_peaks_count))

        if matched_peaks_count > 0 or scoring == ScoringMode.INTENCITY:
            if not found_any or score > max_score:
                found_any = True
                max_score = score
                max_matched_count = matched_peaks_count
                best_matched_peaks = matched_peaks
                best_modifications = hypothesis.modifications

    final_score = max_score if found_any else 0.0

    return MatchResult(
        structure_id=theoretical_spectrum.structure_id,
        spectrum_id=spectrum_id,
        structure_mass=theoretical_spectrum.structure_mass,
        spectrum_mass=precursor_mass,
        tolerance=tolerance,
        matched_peaks_count=max_matched_count,
        theoretical_peaks_count=len(theoretical_spectrum.ions),
        experimental_peaks_count=len(spec_index.sorted_peaks),
        monomer_count=theoretical_spectrum.monomer_count,
        unmodified_monomer_mass_vector=theoretical_spectrum.monomer_mass_vector,
        score=final_score,
        matched_peaks=best_matched_peaks,
        modifications=best_modifications,
    )


def match_spectrum(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    tolerance: Tolerance,
    scoring: ScoringMode,
    mode: ModificationMode = ModificationMode.FREE_MOD,
    spec_index: ExperimentalSpectrumIndex | None = None,
) -> MatchResult:
    """Match an experimental spectrum to a theoretical spectrum according to mode.

    Args:
        spectrum: The experimental Spectrum instance.
        theoretical_spectrum: The TheoreticalSpectrum instance.
        tolerance: The Tolerance configuration allowed for a match.
        scoring: Scoring mode.
        mode: ModificationMode configuration.
        spec_index: Optional pre-built ExperimentalSpectrumIndex for performance.

    Returns:
        A MatchResult instance containing matching metrics.
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
            theoretical_spectrum, mass_mod, x_mode_only=False
        )
    elif mode == ModificationMode.X_MOD:
        hypotheses = _generate_single_mod_hypotheses(
            theoretical_spectrum, mass_mod, x_mode_only=True
        )
    elif mode == ModificationMode.TWO_MOD:
        hypotheses = itertools.chain(
            _generate_single_mod_hypotheses(theoretical_spectrum, mass_mod, x_mode_only=True),
            _generate_two_mod_hypotheses(theoretical_spectrum, mass_mod),
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
        ions = _calculate_theory_ions_with_mass(theo, 1)
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
    mode: ModificationMode = ModificationMode.FREE_MOD,
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
        spec_index = build_experimental_spectrum_index(spec)
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
                        spec_with_charge,
                        theo,
                        tolerance,
                        scoring,
                        mode=mode,
                        spec_index=spec_index,
                    )
                    if result.matched_peaks_count >= min_shared_peaks:
                        results.append(result)

    logger.info(
        "Completed batch matching. Generated %d total MatchResult(s).",
        len(results),
    )
    return results
