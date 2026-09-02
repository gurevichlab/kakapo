from typing import cast

import numpy as np

from kakapo.constants import PROTON_MASS
from kakapo.core.matcher.indexing import (
    ExperimentalSpectrumIndex,
    _build_structure_index,
    _filter_candidates_with_index,
    build_experimental_spectrum_index,
    get_matching_range,
)
from kakapo.models import (
    Chirality,
    Monomer,
    RuleShift,
    Spectrum,
    TheoreticalIon,
    TheoreticalSpectrum,
    Tolerance,
)


def make_spectrum(
    peaks_data: list[tuple[float, float]],
    precursor_mass: float = 500.0,
    charge: int | None = 1,
    spectrum_id: str = "spec_1",
    source_file_path: str = "test.mgf",
) -> Spectrum:
    """Helper to create a Spectrum from list of mass-intensity tuples."""
    mzs = np.array([p[0] for p in peaks_data], dtype=np.float64)
    intensities = np.array([p[1] for p in peaks_data], dtype=np.float64)
    charge_val = charge if charge is not None else 1
    precursor_mz = (precursor_mass + PROTON_MASS * charge_val) / charge_val
    return Spectrum(
        spectrum_id=spectrum_id,
        mzs=mzs,
        intensities=intensities,
        precursor_mass=precursor_mz,
        charge=charge,
        source_file_path=source_file_path,
    )


def make_theoretical_spectrum(
    ion_masses: list[float],
    structure_mass: float = 500.0,
    monomer_count: int = 4,
    structure_id: str = "struct_1",
    monomer_mass: float = 125.0,
) -> TheoreticalSpectrum:
    """Helper to create a TheoreticalSpectrum from list of masses."""
    ions = frozenset(
        TheoreticalIon(
            fragment_id=f"frag_{idx}",
            mask=(1 << idx),
            rules_shifts=(RuleShift("TEST", mass - monomer_mass),),
        )
        for idx, mass in enumerate(ion_masses)
    )
    monomers = {
        idx: Monomer(
            norine_name=f"Mon_{idx}",
            residue="X",
            is_methylated=False,
            chirality=Chirality.L,
            mass=monomer_mass,
        )
        for idx in range(monomer_count)
    }
    return TheoreticalSpectrum(
        structure_id=structure_id,
        ions=ions,
        structure_mass=structure_mass,
        monomer_count=monomer_count,
        monomer_mass_vector=(monomer_mass,) * monomer_count,
        monomers=monomers,
    )


class TestSpectrumMatcherIndexing:
    """Unit tests for the spectrum matching engine indexing and search functionality."""

    def test_build_structure_index_and_filtering(self):
        """Test building structure index and filtering candidate structures."""
        theo1 = make_theoretical_spectrum(
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            structure_id="struct_1",
            monomer_count=6,
        )
        theo2 = make_theoretical_spectrum(
            [200.0, 210.0, 220.0, 230.0, 240.0, 250.0],
            structure_id="struct_2",
            monomer_count=6,
        )

        index = _build_structure_index([theo1, theo2], bin_width=0.2)

        # Verify index properties
        assert index.bin_width == 0.2
        assert len(index.sorted_peaks) == 12
        masses = [p[0] for p in index.sorted_peaks]
        assert masses == sorted(masses)
        assert len(index.index_table) > 0

        # Create spectrum with peaks matching struct_1
        spec_matching_1 = make_spectrum(
            [
                (100.0 + PROTON_MASS, 1.0),
                (110.0 + PROTON_MASS, 1.0),
                (120.0 + PROTON_MASS, 1.0),
                (130.0 + PROTON_MASS, 1.0),
                (140.0 + PROTON_MASS, 1.0),
            ],
            precursor_mass=500.0,
            charge=1,
        )

        # min_shared_peaks=5: Should only return struct_1
        candidates = _filter_candidates_with_index(
            spec_matching_1, index, tolerance=Tolerance(0.01), min_shared_peaks=5
        )
        assert candidates == {"struct_1"}

        # min_shared_peaks=6: spec only has 5 matching peaks, should return empty set
        candidates_6 = _filter_candidates_with_index(
            spec_matching_1, index, tolerance=Tolerance(0.01), min_shared_peaks=6
        )
        assert candidates_6 == set()

        # Empty spectrum validation using a mock object with empty peaks
        class MockSpectrum:
            def __init__(self):
                self.mzs = np.empty(0, dtype=np.float64)
                self.intensities = np.empty(0, dtype=np.float64)
                self.precursor_mass = 500.0
                self.charge = 1

        mock_spec = cast(Spectrum, MockSpectrum())
        assert _filter_candidates_with_index(mock_spec, index, tolerance=Tolerance(0.01)) == set()

    def test_index_bin_border_handling(self):
        """Test that peaks near and on bin borders are correctly indexed and queried."""
        # bin_width = 0.2 Da. Boundaries are 0.0, 0.2, 0.4, ...
        # Place peaks near and on the bin border (0.2)
        theo = make_theoretical_spectrum(
            [0.199, 0.2, 0.201],
            structure_id="struct_borders",
            monomer_count=3,
        )
        index = _build_structure_index([theo], bin_width=0.2)

        # Query peak close to 0.2 (0.205) with charge 1 and tolerance 0.01
        spec = make_spectrum(
            [(0.205 + PROTON_MASS, 1.0)],
            precursor_mass=500.0,
            charge=1,
        )

        candidates = _filter_candidates_with_index(
            spec, index, tolerance=Tolerance(0.01), min_shared_peaks=1
        )
        assert candidates == {"struct_borders"}

    def test_index_query_low_mass_boundaries(self):
        """Test query handling when i_lower calculations result in negative values."""
        theo = make_theoretical_spectrum(
            [0.01],
            structure_id="struct_near_zero",
            monomer_count=1,
        )
        index = _build_structure_index([theo], bin_width=0.2)

        # Query peak close to 0.015, tolerance = 0.02.
        # neutral mass = 0.015, lower bound is -0.005 -> safeguarded to 0.
        spec = make_spectrum(
            [(0.015 + PROTON_MASS, 1.0)],
            precursor_mass=500.0,
            charge=1,
        )

        candidates = _filter_candidates_with_index(
            spec, index, tolerance=Tolerance(0.02), min_shared_peaks=1
        )
        assert candidates == {"struct_near_zero"}

    def test_index_query_high_mass_boundaries(self):
        """Test query handling when i_lower is larger than index table boundaries."""
        theo = make_theoretical_spectrum(
            [10.0],
            structure_id="struct_small",
            monomer_count=1,
        )
        index = _build_structure_index([theo], bin_width=0.2)

        # Query peak with very high mass (e.g. 500.0)
        spec = make_spectrum(
            [(500.0 + PROTON_MASS, 1.0)],
            precursor_mass=500.0,
            charge=1,
        )

        candidates = _filter_candidates_with_index(
            spec, index, tolerance=Tolerance(0.02), min_shared_peaks=1
        )
        assert candidates == set()

    def test_index_empty_scenarios(self):
        """Test index building and querying with empty inputs or empty indices."""
        # Build index from empty iterable
        empty_index = _build_structure_index([], bin_width=0.2)
        assert empty_index.bin_width == 0.2
        assert len(empty_index.sorted_peaks) == 0
        assert empty_index.index_table == [0]

        # Query against empty index
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0)
        assert _filter_candidates_with_index(spec, empty_index, tolerance=Tolerance(0.01)) == set()

    def test_index_query_multiple_charges(self):
        """Test that querying matches correctly under multiple charge states."""
        theo = make_theoretical_spectrum(
            [100.0],
            structure_id="struct_multicharge",
            monomer_count=1,
        )
        index = _build_structure_index([theo], bin_width=0.2)

        # Experimental peak representing charge 2: mz = (100.0 + 2 * PROTON_MASS) / 2
        mz = (100.0 + 2.0 * PROTON_MASS) / 2.0
        spec = make_spectrum(
            [(mz, 1.0)],
            precursor_mass=500.0,
            charge=2,
        )

        candidates = _filter_candidates_with_index(
            spec, index, tolerance=Tolerance(0.01), min_shared_peaks=1
        )
        assert candidates == {"struct_multicharge"}


class TestExperimentalSpectrumIndex:
    """Granular unit tests for ExperimentalSpectrumIndex creation and range querying."""

    def test_spectrum_index_creation_and_ranks(self) -> None:
        """Verify index creation, peak sorting, and precalculated intensity ranks."""
        # 3 peaks out of order with varying intensities
        peaks = [(300.0, 50.0), (100.0, 200.0), (200.0, 100.0)]
        spec = make_spectrum(peaks)
        index = build_experimental_spectrum_index(spec, bin_width=0.5)

        assert isinstance(index, ExperimentalSpectrumIndex)
        assert len(index.mzs) == 3
        # Sorted peaks should be ordered by mass ascending
        assert list(index.mzs) == [100.0, 200.0, 300.0]

        # Intensity ranks: peak at 100.0 (intensity 200) -> rank 0
        # peak at 200.0 (intensity 100) -> rank 1
        # peak at 300.0 (intensity 50) -> rank 2
        assert list(index.ranks) == [0, 1, 2]

    def test_get_matching_range_da_tolerance(self) -> None:
        """Verify get_matching_range returns exact index slice for Da tolerance."""
        peaks = [(100.0, 1.0), (100.01, 1.0), (100.05, 1.0), (200.0, 1.0)]
        spec = make_spectrum(peaks)
        index = build_experimental_spectrum_index(spec, bin_width=0.5)

        # Query 100.0 with 0.02 Da tolerance window [99.98, 100.02]
        tol_da = Tolerance(0.02, unit="da")
        left, right = get_matching_range(index, 100.0, tol_da)
        assert (left, right) == (0, 2)
        assert list(index.mzs[left:right]) == [100.0, 100.01]

    def test_get_matching_range_ppm_tolerance(self) -> None:
        """Verify get_matching_range returns exact index slice for ppm tolerance."""
        # At 500.0 Da, 20 ppm = 0.01 Da window [499.99, 500.01]
        peaks = [(499.98, 1.0), (499.995, 1.0), (500.0, 1.0), (500.005, 1.0), (500.02, 1.0)]
        spec = make_spectrum(peaks)
        index = build_experimental_spectrum_index(spec, bin_width=0.5)

        tol_ppm = Tolerance(20.0, unit="ppm", min_tolerance=0.001)
        left, right = get_matching_range(index, 500.0, tol_ppm)
        assert list(index.mzs[left:right]) == [499.995, 500.0, 500.005]

    def test_get_matching_range_out_of_bounds(self) -> None:
        """Verify out of bounds queries return empty range (0, 0)."""
        peaks = [(100.0, 1.0), (200.0, 1.0)]
        spec = make_spectrum(peaks)
        index = build_experimental_spectrum_index(spec, bin_width=0.5)
        tol = Tolerance(0.01)

        # Query below min mass
        assert get_matching_range(index, 50.0, tol) == (0, 0)
        # Query above max mass
        assert get_matching_range(index, 300.0, tol) == (0, 0)
