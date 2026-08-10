"""Search index structures and binary search/filtering helpers for peak matching."""

import bisect
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from kakapo.config import DEFAULT_MIN_PEAKS
from kakapo.constants import MAX_CHARGE, PROTON_MASS
from kakapo.models import (
    MonomerModification,
    Peak,
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


def _calculate_theory_ions_with_mass(
    theoretical_spectrum: TheoreticalSpectrum,
    max_charge: int,
    modifications: Iterable[MonomerModification] = (),
) -> list[TheoreticalIonWithMass]:
    """Calculate and sort theoretical ion masses based on monomer masses and charges.

    Uses precomputed unmodified masses to update mass in O(1) for modified monomers.

    Args:
        theoretical_spectrum: The TheoreticalSpectrum containing unmodified ions.
        max_charge: The maximum charge state to generate ions for.
        modifications: Active monomer modifications to apply to the masses.

    Returns:
        A list of TheoreticalIonWithMass objects, sorted ascending by m/z.
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
