"""Search index structures and binary search/filtering helpers for peak matching."""

import bisect
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from kakapo.config import DEFAULT_MIN_PEAKS
from kakapo.constants import MAX_CHARGE, PROTON_MASS
from kakapo.models import (
    MonomerModification,
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
        mzs: 1D float64 NumPy array of peak m/z masses (sorted ascending).
        intensities: 1D float64 NumPy array of corresponding peak intensities.
        ranks: 1D int64 NumPy array of intensity ranks (0 = highest intensity).
        bin_width: Width of mass bins in Daltons for O(1) table lookup.
        index_table: 1D int64 NumPy array of start indices for each mass bin.
        mzs_list: Tuple of peak m/z floats for ultra-fast C bisect searches.
    """

    mzs: np.ndarray
    intensities: np.ndarray
    ranks: np.ndarray
    bin_width: float
    index_table: np.ndarray
    mzs_list: tuple[float, ...] = ()


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
    if len(spectrum.mzs) == 0:
        return ExperimentalSpectrumIndex(
            mzs=np.empty(0, dtype=np.float64),
            intensities=np.empty(0, dtype=np.float64),
            ranks=np.empty(0, dtype=np.int64),
            bin_width=bin_width,
            index_table=np.zeros(1, dtype=np.int64),
            mzs_list=(),
        )

    # Ensure mzs and intensities are sorted by m/z
    if not np.all(spectrum.mzs[:-1] <= spectrum.mzs[1:]):
        sort_idx = np.argsort(spectrum.mzs)
        mzs = spectrum.mzs[sort_idx]
        intensities = spectrum.intensities[sort_idx]
    else:
        mzs = spectrum.mzs
        intensities = spectrum.intensities

    ranks = np.argsort(np.argsort(-intensities))

    max_mass = mzs[-1]
    num_bins = int(max_mass // bin_width) + 3

    bin_targets = np.arange(num_bins, dtype=np.float64) * bin_width
    index_table = np.searchsorted(mzs, bin_targets, side="left")

    return ExperimentalSpectrumIndex(
        mzs=mzs,
        intensities=intensities,
        ranks=ranks,
        bin_width=bin_width,
        index_table=index_table,
        mzs_list=tuple(mzs.tolist()),
    )


def get_left_index(index: ExperimentalSpectrumIndex, min_mass: float) -> int:
    """Find the lower bound index in index.mzs for min_mass.

    Args:
        index: The experimental spectrum index.
        min_mass: Minimum mass threshold.

    Returns:
        The first index where peak mass >= min_mass.
    """
    mzs_list = index.mzs_list
    if not mzs_list or min_mass > mzs_list[-1]:
        return len(mzs_list)
    if min_mass <= mzs_list[0]:
        return 0

    i_lower = int(min_mass // index.bin_width)
    if i_lower >= len(index.index_table):
        return len(mzs_list)
    if i_lower < 0:
        i_lower = 0

    start_search = int(index.index_table[i_lower])
    return bisect.bisect_left(mzs_list, min_mass, lo=start_search)


def get_matching_range(
    index: ExperimentalSpectrumIndex, mz: float, tolerance: Tolerance
) -> tuple[int, int]:
    """Find index bounds [start_idx, end_idx) of experimental peaks matching theoretical m/z.

    Args:
        index: The experimental spectrum index.
        mz: Theoretical m/z value.
        tolerance: Tolerance configuration (da or ppm).

    Returns:
        A tuple of (start_idx, end_idx) in index.mzs.
    """
    mzs_list = index.mzs_list
    if not mzs_list:
        return 0, 0

    tol = tolerance.get_limit(mz)
    min_mass = mz - tol
    max_mass = mz + tol

    if max_mass < mzs_list[0] or min_mass > mzs_list[-1]:
        return 0, 0

    left_idx = get_left_index(index, min_mass)

    i_upper = int(max_mass // index.bin_width)
    end_search = index.index_table[min(i_upper + 1, len(index.index_table) - 1)]
    right_idx = bisect.bisect_right(mzs_list, max_mass, lo=left_idx, hi=end_search)

    return left_idx, right_idx


@dataclass
class _StructureIndex:
    """Index structure for fast theoretical spectrum peak lookup.

    Attributes:
        bin_width: Bin width in Daltons.
        sorted_peaks: List of sorted tuples (theoretical_ion_mass, structure_id).
        theo_masses: List of theoretical ion mass floats.
        theo_struct_ids: List of corresponding structure IDs.
        index_table: Mapping from bin index to offset in sorted_peaks.
    """

    bin_width: float
    sorted_peaks: list[tuple[float, str]]
    theo_masses: list[float]
    theo_struct_ids: list[str]
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
        return _StructureIndex(
            bin_width=bin_width,
            sorted_peaks=[],
            theo_masses=[],
            theo_struct_ids=[],
            index_table=[0],
        )

    max_mass = flat_peaks[-1][0]
    num_bins = int(max_mass // bin_width) + 3

    theo_masses = [p[0] for p in flat_peaks]
    theo_struct_ids = [p[1] for p in flat_peaks]

    index_table: list[int] = []
    for i in range(num_bins):
        target = i * bin_width
        idx = bisect.bisect_left(theo_masses, target)
        index_table.append(idx)

    return _StructureIndex(
        bin_width=bin_width,
        sorted_peaks=flat_peaks,
        theo_masses=theo_masses,
        theo_struct_ids=theo_struct_ids,
        index_table=index_table,
    )


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
    if len(spectrum.mzs) == 0 or not index.theo_masses:
        return set()

    max_charge = spectrum.charge if spectrum.charge is not None else MAX_CHARGE
    hits: Counter[str] = Counter()

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
            index.theo_masses,
            target_mass - tolerance,
            lo=start_idx,
            hi=end_idx,
        )
        right_idx = bisect.bisect_right(
            index.theo_masses,
            target_mass + tolerance,
            lo=start_idx,
            hi=end_idx,
        )

        for idx in range(left_idx, right_idx):
            hits[index.theo_struct_ids[idx]] += 1

    charges = np.arange(1, max_charge + 1)
    mzs = spectrum.mzs

    neutral_matrix = mzs[:, None] * charges - PROTON_MASS * charges

    for peak_idx in range(len(mzs)):
        peak_mz = mzs[peak_idx]
        tol_mz = tolerance.get_limit(peak_mz)

        for c_idx, c in enumerate(charges):
            neutral_mass = neutral_matrix[peak_idx, c_idx]
            neutral_tolerance = c * tol_mz

            # 1. Unmodified peak match
            query_index(neutral_mass, neutral_tolerance)

            # 2. Modified peak match (via its unmodified complement fragment)
            precursor_neutral_c = spectrum.precursor_mass * c - PROTON_MASS * c
            complement_mass = precursor_neutral_c - neutral_mass
            if complement_mass > neutral_tolerance:
                query_index(complement_mass, neutral_tolerance)

    return {struct_id for struct_id, count in hits.items() if count >= min_shared_peaks}


def _calculate_theory_ions_with_mass(
    theoretical_spectrum: TheoreticalSpectrum,
    max_charge: int,
    modifications: Iterable[MonomerModification] = (),
) -> list[TheoreticalIonWithMass]:
    """Calculate and sort theoretical ion masses based on monomer masses and charges.

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
