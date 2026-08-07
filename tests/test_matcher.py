from dataclasses import replace
from typing import cast

import pytest

from kakapo.constants import PROTON_MASS
from kakapo.core.matcher import (
    ExperimentalSpectrumIndex,
    _build_structure_index,
    _filter_candidates_with_index,
    build_experimental_spectrum_index,
    get_matching_range,
    match_spectra,
    match_spectrum,
)
from kakapo.models import (
    Chirality,
    ModificationMode,
    Monomer,
    Peak,
    RuleShift,
    ScoringMode,
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
    peaks = frozenset(Peak(mass=m, intensity=i) for m, i in peaks_data)
    charge_val = charge if charge is not None else 1
    precursor_mz = (precursor_mass + PROTON_MASS * charge_val) / charge_val
    return Spectrum(
        spectrum_id=spectrum_id,
        peaks=peaks,
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


class TestSpectrumMatcher:
    """Unit tests for the spectrum matching engine."""

    def test_basic_matching_exact(self):
        """Test basic peak matching where peaks match the theoretical ions exactly."""
        peaks = [(100.0, 1.0), (200.0, 1.0), (300.0, 1.0)]
        ion_masses = [100.0 - 1.007276, 200.0 - 1.007276, 300.0 - 1.007276]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.ALL_PEAKS
        )
        assert result.matched_peaks_count == 3

    def test_basic_matching_within_tolerance(self):
        """Test basic peak matching where peaks are slightly off but within the tolerance window."""
        peaks = [(100.01, 1.0), (200.02, 1.0)]
        ion_masses = [100.0 - 1.007276, 200.0 - 1.007276]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.03), scoring=ScoringMode.ALL_PEAKS
        )
        assert result.matched_peaks_count == 2

    def test_basic_matching_outside_tolerance(self):
        """Test basic peak matching where peaks are outside the tolerance window."""
        peaks = [(100.01, 1.0), (200.02, 1.0)]
        ion_masses = [100.0 - 1.007276, 200.0 - 1.007276]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.005), scoring=ScoringMode.ALL_PEAKS
        )
        assert result.matched_peaks_count == 0

    def test_basic_matching_no_matches(self):
        """Test basic peak matching where theoretical ions do not match any peaks."""
        peaks = [(150.0, 1.0), (250.0, 1.0)]
        ion_masses = [100.0 - 1.007276, 200.0 - 1.007276]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(spec, theo, tolerance=Tolerance(0.1), scoring=ScoringMode.ALL_PEAKS)
        assert result.matched_peaks_count == 0

    def test_greedy_matching_unique_peaks(self):
        """Ensure each experimental peak can match at most one theoretical ion, and vice versa."""
        # Two theoretical ions close to the same experimental peak
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [99.99 - 1.007276, 100.01 - 1.007276],
            structure_mass=500.0,
            monomer_count=2,
        )

        # Only 1 match should occur because there is only 1 experimental peak
        result1 = match_spectrum(
            spec, theo, tolerance=Tolerance(0.05), scoring=ScoringMode.ALL_PEAKS
        )
        assert result1.matched_peaks_count == 1

        # Two experimental peaks close to the same theoretical ion
        spec2 = make_spectrum([(99.99, 1.0), (100.01, 1.0)], precursor_mass=500.0)
        theo2 = make_theoretical_spectrum([100.0 - 1.007276], structure_mass=500.0, monomer_count=1)

        # Only 1 match should occur because there is only 1 theoretical ion
        result2 = match_spectrum(
            spec2, theo2, tolerance=Tolerance(0.05), scoring=ScoringMode.ALL_PEAKS
        )
        assert result2.matched_peaks_count == 1

    def test_greedy_matching_takes_first(self):
        """Ensure greedy matching takes the first available peaks."""
        spec = make_spectrum([(100.01, 1.0), (100.05, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276, 100.04 - 1.007276],
            structure_mass=500.0,
            monomer_count=2,
        )

        # 100.0 matches 100.01 (first)
        # 100.04 matches 100.05 (remaining)
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.02), scoring=ScoringMode.ALL_PEAKS
        )
        assert result.matched_peaks_count == 2

    def test_matched_peaks_populated(self):
        """Verify matched_peaks details are populated in MatchResult."""
        spec = make_spectrum([(100.01, 1.0), (200.0, 1.0), (300.05, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276, 150.0 - 1.007276, 300.04 - 1.007276],
            structure_mass=500.0,
            monomer_count=3,
        )

        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.02), scoring=ScoringMode.ALL_PEAKS
        )
        assert result.matched_peaks_count == 2
        assert result.experimental_peaks_count == 3
        assert len(result.matched_peaks) == 2

        # Check first matched peak
        ep1 = result.matched_peaks[0]
        assert ep1.theoretical_mz == pytest.approx(100.0)
        assert ep1.experimental_mz == pytest.approx(100.01)
        assert ep1.charge == 1

        # Check second matched peak
        ep2 = result.matched_peaks[1]
        assert ep2.theoretical_mz == pytest.approx(300.04)
        assert ep2.experimental_mz == pytest.approx(300.05)
        assert ep2.charge == 1

    @pytest.mark.parametrize(
        "spec_charge,expected_count,expected_charge",
        [
            (1, 0, None),
            (2, 1, 2),
            (None, 1, 2),  # Fallback to MAX_CHARGE = 2
        ],
    )
    def test_charge_matching_limits(self, spec_charge, expected_count, expected_charge):
        """Test matching charge limit restrictions and fallback behavior."""
        theo = make_theoretical_spectrum([198.0], structure_mass=500.0, monomer_count=1)
        spec = make_spectrum([(100.007276, 1.0)], precursor_mass=500.0, charge=spec_charge)
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.ALL_PEAKS
        )
        assert result.matched_peaks_count == expected_count
        if expected_count > 0:
            assert result.matched_peaks[0].charge == expected_charge

    def test_matching_multiple_charges(self):
        """Test matching peaks of different charges when spectrum charge allows it."""
        theo = make_theoretical_spectrum([198.0, 298.0], structure_mass=500.0, monomer_count=2)
        spec = make_spectrum([(199.007276, 1.0), (150.007276, 1.0)], precursor_mass=500.0, charge=2)
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.ALL_PEAKS
        )

        assert result.matched_peaks_count == 2

        # Sort matched peaks by charge for deterministic assertions
        sorted_matched = sorted(result.matched_peaks, key=lambda p: p.charge)

        # Check first matched peak (charge 1)
        assert sorted_matched[0].charge == 1
        assert sorted_matched[0].theoretical_mz == pytest.approx(199.007276)
        assert sorted_matched[0].experimental_mz == pytest.approx(199.007276)

        # Check second matched peak (charge 2)
        assert sorted_matched[1].charge == 2
        assert sorted_matched[1].theoretical_mz == pytest.approx(150.007276)
        assert sorted_matched[1].experimental_mz == pytest.approx(150.007276)

    def test_match_spectra_pairwise(self):
        """Test matching lists of experimental and theoretical spectra pairwise."""
        spec1 = make_spectrum([(100.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_1")
        spec2 = make_spectrum([(200.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_2")

        theo1 = make_theoretical_spectrum(
            [100.0 - 1.007276], structure_mass=500.0, monomer_count=2, structure_id="struct_1"
        )
        theo2 = make_theoretical_spectrum(
            [200.0 - 1.007276], structure_mass=500.0, monomer_count=2, structure_id="struct_2"
        )
        results = match_spectra(
            [spec1, spec2],
            [theo1, theo2],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            min_shared_peaks=1,
        )

        assert len(results) == 2

        # Key results by (spectrum_id, structure_id)
        results_map = {(r.spectrum_id, r.structure_id): r for r in results}
        assert results_map[("spec_1", "struct_1")].matched_peaks_count == 1
        assert results_map[("spec_2", "struct_2")].matched_peaks_count == 1

    def test_match_spectra_unknown_charge(self):
        """Test matching spectrum with unknown charge (charge=None)."""
        spec = make_spectrum(
            [(500.0, 100.0)], precursor_mass=1000.0, charge=None, spectrum_id="spec_unknown_charge"
        )
        true_precursor_mz = (1000.0 + 2 * PROTON_MASS) / 2

        spec = replace(spec, precursor_mass=true_precursor_mz)

        theo = make_theoretical_spectrum(
            [500.0 - PROTON_MASS], structure_mass=1000.0, monomer_count=2, structure_id="struct_1"
        )

        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            min_shared_peaks=1,
        )

        assert len(results) == 1
        assert results[0].spectrum_id == "spec_unknown_charge"
        assert results[0].spectrum_mass == true_precursor_mz
        assert results[0].matched_peaks_count == 1
        assert results[0].matched_peaks[0].charge == 1

    def test_match_spectra_empty_inputs(self):
        """Test match_spectra with empty input iterables."""
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum([100.0 - 1.007276], structure_mass=500.0, monomer_count=1)

        assert (
            len(match_spectra([], [theo], tolerance=Tolerance(0.01), scoring=ScoringMode.ALL_PEAKS))
            == 0
        )
        assert (
            len(match_spectra([spec], [], tolerance=Tolerance(0.01), scoring=ScoringMode.ALL_PEAKS))
            == 0
        )
        assert (
            len(match_spectra([], [], tolerance=Tolerance(0.01), scoring=ScoringMode.ALL_PEAKS))
            == 0
        )

    @pytest.mark.parametrize(
        "structure_mass,expected_match",
        [
            # Upper bound checks (precursor = 500.0, diff <= 250.0 -> max structure_mass = 750.0)
            (749.99, True),
            (750.0, True),
            (750.01, False),
            # Lower bound checks (precursor = 500.0, diff <= 250.0 -> min structure_mass = 250.0)
            (250.01, True),
            (250.0, True),
            (249.99, False),
        ],
    )
    def test_match_spectra_mass_difference_limits(self, structure_mass, expected_match):
        """Test pairwise matching filters pairs outside MAX_TOTAL_MASS_DIFFERENCE constraint."""
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_1")
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276],
            structure_mass=structure_mass,
            structure_id="struct_test",
            monomer_mass=500.0,
        )
        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            min_shared_peaks=1,
        )
        if expected_match:
            assert len(results) == 1
            assert results[0].structure_id == "struct_test"
        else:
            assert len(results) == 0

    def test_match_spectra_zero_peaks_skipped(self):
        """Test that pairwise matching skips pairs with 0 matched peaks."""
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_1")
        # Structure mass = 700.0, meaning mass_mod = -200.0.
        # Since monomer_mass is only 125.0, this modification is invalid (<=0),
        # so match_spectrum will yield 0 matched peaks.
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276],
            structure_mass=700.0,
            structure_id="struct_zero_peaks",
            monomer_mass=125.0,
        )
        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            min_shared_peaks=1,
        )
        assert len(results) == 0

    def test_match_spectra_negative_monomer_modification_skipped(self):
        """Test that pairwise matching filters out structures where max monomer mass plus
        mass difference is negative.
        """
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_1")
        # Structure mass is 700.0 -> mass_mod = 500.0 - 700.0 = -200.0.
        # Monomer mass is 150.0 -> max is 150.0. 150.0 + (-200.0) = -50.0 < 0.
        # This structure should be skipped.
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276],
            structure_mass=700.0,
            structure_id="struct_neg_monomer",
            monomer_mass=150.0,
        )
        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            min_shared_peaks=1,
        )
        assert len(results) == 0

    def test_match_spectra_zero_mass_modification_allowed(self):
        """Test that a mass modification resulting in exactly 0 monomer mass is allowed
        and matched.
        """
        # Precursor neutral mass = 300.0 (m/z ~ 301.007)
        # Structure mass = 500.0. Mass mod = -200.0.
        # Monomer 0 mass = 200.0 -> updated to 200.0 - 200.0 = 0.0 (allowed).
        # Monomer 1 mass = 300.0 -> remains 300.0.
        # Theoretical ion: fragment contains Monomer 1 (mass 300.0).
        # Experimental peak: 300.0 + PROTON_MASS = 301.007276.
        spec = make_spectrum([(301.007276, 1.0)], precursor_mass=300.0, spectrum_id="spec_1")

        # Build theoretical spectrum
        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_0",
                    mask=2,  # Monomer 1 only
                    rules_shifts=(RuleShift("TEST", 0.0),),
                )
            ]
        )
        monomers = {
            0: Monomer(
                norine_name="Mon_0",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=200.0,
            ),
            1: Monomer(
                norine_name="Mon_1",
                residue="Y",
                is_methylated=False,
                chirality=Chirality.L,
                mass=300.0,
            ),
        }
        theo = TheoreticalSpectrum(
            structure_id="struct_zero_mass",
            ions=ions,
            structure_mass=500.0,
            monomer_count=2,
            monomer_mass_vector=(200.0, 300.0),
            monomers=monomers,
        )

        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            min_shared_peaks=1,
        )
        assert len(results) == 1
        assert results[0].matched_peaks_count == 1
        assert len(results[0].modifications) == 1
        assert results[0].modifications[0].monomer_idx == 0

    def test_match_spectra_min_shared_peaks_filtering(self):
        """Verify that match_spectra filters out results where matched_peaks_count < min_shared_peaks."""
        theo = make_theoretical_spectrum(
            [100.0 - PROTON_MASS, 200.0 - PROTON_MASS],
            structure_id="struct_min_shared",
            structure_mass=300.0,
            monomer_count=2,
        )

        spec = make_spectrum(
            [(100.0, 1.0), (999.0, 1.0)],
            precursor_mass=300.0 + PROTON_MASS,
            charge=1,
            spectrum_id="spec_min_shared",
        )

        # min_shared_peaks=1 -> match returned
        results_1 = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            min_shared_peaks=1,
        )
        assert len(results_1) == 1
        assert results_1[0].matched_peaks_count == 1

        # min_shared_peaks=2 -> match filtered out
        results_2 = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            min_shared_peaks=2,
        )
        assert len(results_2) == 0

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
                self.peaks = frozenset()
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

    @pytest.mark.parametrize(
        "monomer_masses,precursor_mass,peaks,expected_matched,expected_mod,expected_idx",
        [
            # Positive shift (monomer 1 modified: 150.0 + 30.0 = 180.0)
            (
                (100.0, 150.0, 200.0),
                480.0,
                [(100.0 + 1.007276, 1.0), (180.0 + 1.007276, 1.0), (200.0 + 1.007276, 1.0)],
                3,
                30.0,
                1,
            ),
            # Negative shift (monomer 2 modified: 200.0 - 20.0 = 180.0)
            (
                (100.0, 150.0, 200.0),
                430.0,
                [(100.0 + 1.007276, 1.0), (150.0 + 1.007276, 1.0), (180.0 + 1.007276, 1.0)],
                3,
                -20.0,
                2,
            ),
            # Zero shift (no modifications, modified_monomer_idx is None)
            (
                (100.0, 150.0, 200.0),
                450.0,
                [(100.0 + 1.007276, 1.0), (150.0 + 1.007276, 1.0), (200.0 + 1.007276, 1.0)],
                3,
                0.0,
                None,
            ),
            # Single monomer
            (
                (150.0,),
                160.0,
                [(160.0 + 1.007276, 1.0)],
                1,
                10.0,
                0,
            ),
            # Non-positive monomer skipped (monomer 0 becomes 15 - 20 = -5 <= 0, monomer 1 modified:
            # 150 - 20 = 130)
            (
                (15.0, 150.0),
                145.0,
                [(15.0 + 1.007276, 1.0), (130.0 + 1.007276, 1.0), (145.0 + 1.007276, 1.0)],
                3,
                -20.0,
                1,
            ),
        ],
    )
    def test_modification_tolerant_matching_scenarios(
        self, monomer_masses, precursor_mass, peaks, expected_matched, expected_mod, expected_idx
    ):
        """Test various modification-tolerant matching scenarios including shift types and edge cases."""
        ions = frozenset(
            TheoreticalIon(fragment_id=f"frag_{idx}", mask=1 << idx)
            for idx in range(len(monomer_masses))
        )
        if len(monomer_masses) == 2:
            ions = frozenset(list(ions) + [TheoreticalIon(fragment_id="frag_2", mask=3)])

        monomers = {
            idx: Monomer(
                norine_name=f"Mon_{idx}",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=mass,
            )
            for idx, mass in enumerate(monomer_masses)
        }
        theo = TheoreticalSpectrum(
            structure_id="struct_mod",
            ions=ions,
            structure_mass=sum(monomer_masses),
            monomer_count=len(monomer_masses),
            monomer_mass_vector=monomer_masses,
            monomers=monomers,
        )

        spec = make_spectrum(peaks, precursor_mass=precursor_mass, charge=1)
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.ALL_PEAKS
        )
        assert result.matched_peaks_count == expected_matched
        if expected_idx is None:
            assert len(result.modifications) == 0
        else:
            assert len(result.modifications) == 1
            mod = result.modifications[0]
            assert mod.mass_difference == pytest.approx(expected_mod)
            assert mod.monomer_idx == expected_idx
            assert mod.initial_monomer_mass == monomer_masses[expected_idx]

    def test_x_mode_matching_unambiguous(self) -> None:
        """Verify that x_mode=True only permits modifications on monomers starting with 'X'."""
        monomers = {
            0: Monomer(
                norine_name="X0",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
            1: Monomer(
                norine_name="Ala",
                residue="A",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
        }

        # frag_0 has a shift of +5.0 (mass 105.0 unmodified)
        # frag_1 has a shift of 0.0 (mass 100.0 unmodified)
        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_0", mask=1, rules_shifts=(RuleShift("PEPTIDE", 5.0),)
                ),
                TheoreticalIon(fragment_id="frag_1", mask=2),
            ]
        )

        theo = TheoreticalSpectrum(
            structure_id="struct_x",
            ions=ions,
            structure_mass=205.0,  # sum of unmodified monomers (200.0) + shift (5.0)
            monomer_count=2,
            monomer_mass_vector=(100.0, 100.0),
            monomers=monomers,
        )

        # Precursor neutral mass = 215.0 (+10.0 mod)
        # If we modify X0 (idx 0), we expect:
        # frag_0 -> 115.0
        # frag_1 -> 100.0
        #
        # If we modify Ala (idx 1), we expect:
        # frag_0 -> 105.0
        # frag_1 -> 110.0

        # Let's test experimental spectrum that has modified Ala (110.0)
        # This requires modifying Ala (idx 1) to match.
        spec_ala = Spectrum(
            spectrum_id="spec_ala",
            source_file_path="test.mgf",
            peaks=frozenset([Peak(110.0 + 1.0078, 1.0)]),
            precursor_mass=215.0 + 1.0078,
            charge=1,
        )

        # In FREE_MOD mode, it should successfully match (modifying Ala, idx 1, to get peak 110.0)
        res_normal = match_spectrum(
            spec_ala,
            theo,
            tolerance=Tolerance(0.02),
            scoring=ScoringMode.ALL_PEAKS,
            mode=ModificationMode.FREE_MOD,
        )
        assert res_normal.matched_peaks_count == 1
        assert len(res_normal.modifications) == 1
        assert res_normal.modifications[0].monomer_idx == 1

        # In X_MOD mode, it should NOT match since Ala doesn't start with X,
        # and modifying X0 (idx 0) doesn't produce peak 110.0 (it produces 115.0 and 100.0)
        res_x = match_spectrum(
            spec_ala,
            theo,
            tolerance=Tolerance(0.02),
            scoring=ScoringMode.ALL_PEAKS,
            mode=ModificationMode.X_MOD,
        )
        assert res_x.matched_peaks_count == 0
        assert len(res_x.modifications) == 0

    def test_no_mod_mode(self) -> None:
        """Verify that NO_MOD mode skips modification search and returns 0 matches when precursor mass differs."""
        monomers = {
            0: Monomer(
                norine_name="Ala",
                residue="A",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
        }
        ions = frozenset([TheoreticalIon(fragment_id="frag_0", mask=1)])
        theo = TheoreticalSpectrum(
            structure_id="struct_nomod",
            ions=ions,
            structure_mass=100.0,
            monomer_count=1,
            monomer_mass_vector=(100.0,),
            monomers=monomers,
        )

        # Spectrum with precursor mass = 120.0 (mass_mod = +20.0 > tolerance)
        spec = Spectrum(
            spectrum_id="spec_nomod",
            source_file_path="test.mgf",
            peaks=frozenset([Peak(120.0 + 1.0078, 1.0)]),
            precursor_mass=120.0 + 1.0078,
            charge=1,
        )

        res_no_mod = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.02),
            scoring=ScoringMode.ALL_PEAKS,
            mode=ModificationMode.NO_MOD,
        )
        assert res_no_mod.matched_peaks_count == 0
        assert len(res_no_mod.modifications) == 0

        res_free_mod = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.02),
            scoring=ScoringMode.ALL_PEAKS,
            mode=ModificationMode.FREE_MOD,
        )
        assert res_free_mod.matched_peaks_count == 1
        assert len(res_free_mod.modifications) == 1

    def test_scoring_modes(self):
        """Test all-peaks and exp-peaks scoring modes."""
        # Create a spectrum with a single peak at 100.0, max charge 2
        spec = make_spectrum(
            [(100.0, 1.0)],
            precursor_mass=250.0,
            charge=2,
            spectrum_id="spec_test",
        )

        # Create theoretical spectrum with two ions that will both match 100.0
        # under charge 1 and charge 2 respectively.
        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_1",
                    mask=1,
                    rules_shifts=(RuleShift("TEST", 100.0 - PROTON_MASS - 125.0),),
                ),
                TheoreticalIon(
                    fragment_id="frag_2",
                    mask=2,
                    rules_shifts=(RuleShift("TEST", 200.0 - 2 * PROTON_MASS - 125.0),),
                ),
            ]
        )
        monomers = {
            0: Monomer(
                norine_name="Mon_0",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=125.0,
            ),
            1: Monomer(
                norine_name="Mon_1",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=125.0,
            ),
        }
        theo = TheoreticalSpectrum(
            structure_id="struct_test",
            ions=ions,
            structure_mass=250.0,
            monomer_count=2,
            monomer_mass_vector=(125.0, 125.0),
            monomers=monomers,
        )

        # 1. Test "all-peaks" mode (default)
        res_all = match_spectrum(
            spec, theo, tolerance=Tolerance(0.02), scoring=ScoringMode.ALL_PEAKS
        )
        assert res_all.matched_peaks_count == 2
        assert len(res_all.matched_peaks) == 2
        assert res_all.score == 2

        # 2. Test "exp-peaks" mode
        res_exp = match_spectrum(
            spec, theo, tolerance=Tolerance(0.02), scoring=ScoringMode.EXP_PEAKS
        )
        assert res_exp.matched_peaks_count == 2
        assert len(res_exp.matched_peaks) == 2
        assert res_exp.score == 1

    def test_ppm_tolerance_matching(self):
        """Test matching using ppm relative tolerance."""
        # 20 ppm at 100 Da = 0.002 Da.
        # 20 ppm at 1000 Da = 0.02 Da.

        # Test case 1: peak within ppm window
        # mz = 100.0, experimental peak = 100.0015 (diff = 0.0015 <= 0.002)
        spec1 = make_spectrum([(100.0015, 1.0)], precursor_mass=500.0)
        theo1 = make_theoretical_spectrum(
            [100.0 - PROTON_MASS], structure_mass=500.0, monomer_count=1
        )

        # PPM tolerance = 20.0, min_tolerance = 0.0001 (so ppm limit 0.002 takes precedence)
        tol_ppm = Tolerance(value=20.0, unit="ppm", min_tolerance=0.0001)
        res1 = match_spectrum(spec1, theo1, tolerance=tol_ppm, scoring=ScoringMode.ALL_PEAKS)
        assert res1.matched_peaks_count == 1

        # Test case 2: peak outside ppm window but within min_tolerance
        # mz = 100.0, experimental peak = 100.005 (diff = 0.005 > 0.002)
        # Without min_tolerance it should fail. With min_tolerance = 0.01 it should succeed.
        spec2 = make_spectrum([(100.005, 1.0)], precursor_mass=500.0)

        tol_ppm_low_min = Tolerance(value=20.0, unit="ppm", min_tolerance=0.001)
        res2_fail = match_spectrum(
            spec2, theo1, tolerance=tol_ppm_low_min, scoring=ScoringMode.ALL_PEAKS
        )
        assert res2_fail.matched_peaks_count == 0

        tol_ppm_high_min = Tolerance(value=20.0, unit="ppm", min_tolerance=0.01)
        res2_pass = match_spectrum(
            spec2, theo1, tolerance=tol_ppm_high_min, scoring=ScoringMode.ALL_PEAKS
        )
        assert res2_pass.matched_peaks_count == 1

    def test_intencity_scoring_single_matched_peak(self):
        """Test intensity scoring with a single matched peak."""
        # 1 experimental peak
        peaks = [(100.0, 100.0)]
        # 1 matching theoretical ion (mz = 100.0 at charge 1)
        ion_masses = [100.0 - PROTON_MASS]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.INTENCITY
        )
        assert result.matched_peaks_count == 1
        assert result.score == pytest.approx(3.846)

    def test_intencity_scoring_multiple_ranked_peaks(self):
        """Test intensity scoring with multiple matched peaks of different ranks."""
        peaks2 = [(100.0, 100.0), (200.0, 50.0)]
        ion_masses2 = [100.0 - PROTON_MASS, 200.0 - PROTON_MASS]
        spec2 = make_spectrum(peaks2, precursor_mass=500.0)
        theo2 = make_theoretical_spectrum(
            ion_masses2, structure_mass=500.0, monomer_count=len(ion_masses2)
        )
        result2 = match_spectrum(
            spec2, theo2, tolerance=Tolerance(0.01), scoring=ScoringMode.INTENCITY
        )
        assert result2.matched_peaks_count == 2
        assert result2.score == pytest.approx(3.846 + 4.059)

    def test_intencity_scoring_priority(self):
        """Test charge priority selection (charge 1 over charge 2) in representative matching."""
        peaks = [(100.0, 100.0)]
        # Two theoretical ions that match 100.0 at charge 1 and charge 2 respectively
        ion_masses = [100.0 - PROTON_MASS, 200.0 - 2 * PROTON_MASS]
        spec = make_spectrum(peaks, precursor_mass=500.0, charge=2)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.INTENCITY
        )
        assert result.matched_peaks_count == 1
        assert result.score == pytest.approx(3.203)
        assert result.matched_peaks[0].charge == 1

    def test_intencity_scoring_missing_group_clustered(self):
        """Test missing peak clustering under 2 * tolerance distance constraint."""
        # Tolerance = 0.02. 2 * tolerance = 0.04.
        tol = Tolerance(0.02)
        # Experimental peak far away so nothing matches
        peaks = [(1000.0, 1.0)]
        spec = make_spectrum(peaks, precursor_mass=1000.0, charge=1)

        # Within 2 * tolerance -> single missing group
        ion_masses_clustered = [300.0 - PROTON_MASS, 300.01 - PROTON_MASS]
        theo_clustered = make_theoretical_spectrum(
            ion_masses_clustered, structure_mass=1000.0, monomer_count=2
        )
        res1 = match_spectrum(spec, theo_clustered, tolerance=tol, scoring=ScoringMode.INTENCITY)
        assert res1.matched_peaks_count == 0
        assert res1.score == pytest.approx(-0.337)

    def test_intencity_scoring_missing_groups_separated(self):
        """Test that missing peaks outside 2 * tolerance are penalized as separate groups."""
        tol = Tolerance(0.02)
        peaks = [(1000.0, 1.0)]
        spec = make_spectrum(peaks, precursor_mass=1000.0, charge=1)

        # Outside 2 * tolerance -> separate missing groups
        ion_masses_separated = [300.0 - PROTON_MASS, 300.1 - PROTON_MASS]
        theo_separated = make_theoretical_spectrum(
            ion_masses_separated, structure_mass=1000.0, monomer_count=2
        )
        res2 = match_spectrum(spec, theo_separated, tolerance=tol, scoring=ScoringMode.INTENCITY)
        assert res2.matched_peaks_count == 0
        assert res2.score == pytest.approx(-0.337 * 2)

    def test_two_mod_matching_success(self) -> None:
        """Test two-modification matching with a known base monomer modification and X monomer residual shift."""
        monomers = {
            0: Monomer(
                norine_name="Ala",
                residue="A",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
            1: Monomer(
                norine_name="Xle",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=120.0,
            ),
        }

        # Unmodified structure mass = 220.0
        # Known mod on Ala: NAc-Ala (+C2H2O, mass_difference = 42.010565)
        # Expected Ala mass = 142.010565
        # Desired residual shift on Xle = +20.0 -> Xle mass = 140.0
        # Precursor neutral mass = 220.0 + 42.010565 + 20.0 = 282.010565

        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_0", mask=1, rules_shifts=(RuleShift("PEPTIDE", 0.0),)
                ),
                TheoreticalIon(
                    fragment_id="frag_1", mask=2, rules_shifts=(RuleShift("PEPTIDE", 0.0),)
                ),
            ]
        )

        theo = TheoreticalSpectrum(
            structure_id="struct_two_mod",
            ions=ions,
            structure_mass=220.0,
            monomer_count=2,
            monomer_mass_vector=(100.0, 120.0),
            monomers=monomers,
        )

        # Experimental spectrum with peaks matching 142.010565 + 1.007276 and 140.0 + 1.007276
        spec_peaks = frozenset(
            [
                Peak(mass=142.010565 + 1.007276, intensity=100.0),
                Peak(mass=140.0 + 1.007276, intensity=100.0),
            ]
        )
        spec = Spectrum(
            spectrum_id="spec_two_mod",
            source_file_path="test.mgf",
            peaks=spec_peaks,
            precursor_mass=282.010565 + 1.007276,
            charge=1,
        )

        result = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            mode=ModificationMode.TWO_MOD,
        )

        assert result.matched_peaks_count == 2
        assert len(result.modifications) == 2

        mod1 = result.modifications[0]
        assert mod1.monomer_idx == 0
        assert mod1.initial_monomer_name == "Ala"
        assert mod1.mass_difference == pytest.approx(42.010565)
        assert mod1.new_monomer_name == "NAc-Ala"
        assert mod1.formula_difference == "+C2H2O"
        assert mod1.initial_monomer_mass == 100.0

        mod2 = result.modifications[1]
        assert mod2.monomer_idx == 1
        assert mod2.initial_monomer_name == "Xle"
        assert mod2.mass_difference == pytest.approx(20.0)
        assert mod2.new_monomer_name is None
        assert mod2.initial_monomer_mass == 120.0

    def test_two_mod_matching_no_x_monomers(self) -> None:
        """Verify that two-modification mode produces 0 matches if no X monomers exist in structure."""
        monomers = {
            0: Monomer(
                norine_name="Ala",
                residue="A",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
            1: Monomer(
                norine_name="Val",
                residue="V",
                is_methylated=False,
                chirality=Chirality.L,
                mass=120.0,
            ),
        }

        ions = frozenset([TheoreticalIon(fragment_id="frag_0", mask=1)])
        theo = TheoreticalSpectrum(
            structure_id="struct_no_x",
            ions=ions,
            structure_mass=220.0,
            monomer_count=2,
            monomer_mass_vector=(100.0, 120.0),
            monomers=monomers,
        )

        spec = Spectrum(
            spectrum_id="spec_test",
            source_file_path="test.mgf",
            peaks=frozenset([Peak(mass=150.0, intensity=10.0)]),
            precursor_mass=280.0,
            charge=1,
        )

        result = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            mode=ModificationMode.TWO_MOD,
        )

        assert result.matched_peaks_count == 0
        assert len(result.modifications) == 0

    def test_two_mod_matching_includes_single_mod(self) -> None:
        """Verify that TWO_MOD mode can match a single-modification hypothesis if optimal."""
        monomers = {
            0: Monomer(
                norine_name="Xle",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
            1: Monomer(
                norine_name="Val",
                residue="V",
                is_methylated=False,
                chirality=Chirality.L,
                mass=120.0,
            ),
        }

        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_0", mask=1, rules_shifts=(RuleShift("PEPTIDE", 0.0),)
                )
            ]
        )
        theo = TheoreticalSpectrum(
            structure_id="struct_single_mod_in_two_mod",
            ions=ions,
            structure_mass=220.0,
            monomer_count=2,
            monomer_mass_vector=(100.0, 120.0),
            monomers=monomers,
        )

        # Precursor neutral mass = 220.0 + 50.0 = 270.0
        # Peak matching Ala (+50.0) + proton = 150.0 + 1.007276
        spec = Spectrum(
            spectrum_id="spec_test",
            source_file_path="test.mgf",
            peaks=frozenset([Peak(mass=150.0 + PROTON_MASS, intensity=100.0)]),
            precursor_mass=270.0 + PROTON_MASS,
            charge=1,
        )

        result = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.ALL_PEAKS,
            mode=ModificationMode.TWO_MOD,
        )

        assert result.matched_peaks_count == 1
        assert len(result.modifications) == 1
        assert result.modifications[0].monomer_idx == 0
        assert result.modifications[0].mass_difference == pytest.approx(50.0)


class TestExperimentalSpectrumIndex:
    """Granular unit tests for ExperimentalSpectrumIndex creation and range querying."""

    def test_spectrum_index_creation_and_ranks(self) -> None:
        """Verify index creation, peak sorting, and precalculated intensity ranks."""
        # 3 peaks out of order with varying intensities
        peaks = [(300.0, 50.0), (100.0, 200.0), (200.0, 100.0)]
        spec = make_spectrum(peaks)
        index = build_experimental_spectrum_index(spec, bin_width=0.5)

        assert isinstance(index, ExperimentalSpectrumIndex)
        assert len(index.sorted_peaks) == 3
        # Sorted peaks should be ordered by mass ascending
        assert [p.mass for p in index.sorted_peaks] == [100.0, 200.0, 300.0]
        assert index.masses == (100.0, 200.0, 300.0)

        # Intensity ranks: peak at 100.0 (intensity 200) -> rank 0
        # peak at 200.0 (intensity 100) -> rank 1
        # peak at 300.0 (intensity 50) -> rank 2
        assert index.ranks == (0, 1, 2)

    def test_get_matching_range_da_tolerance(self) -> None:
        """Verify get_matching_range returns exact index slice for Da tolerance."""
        peaks = [(100.0, 1.0), (100.01, 1.0), (100.05, 1.0), (200.0, 1.0)]
        spec = make_spectrum(peaks)
        index = build_experimental_spectrum_index(spec, bin_width=0.5)

        # Query 100.0 with 0.02 Da tolerance window [99.98, 100.02]
        tol_da = Tolerance(0.02, unit="da")
        left, right = get_matching_range(index, 100.0, tol_da)
        assert (left, right) == (0, 2)
        assert [p.mass for p in index.sorted_peaks[left:right]] == [100.0, 100.01]

    def test_get_matching_range_ppm_tolerance(self) -> None:
        """Verify get_matching_range returns exact index slice for ppm tolerance."""
        # At 500.0 Da, 20 ppm = 0.01 Da window [499.99, 500.01]
        peaks = [(499.98, 1.0), (499.995, 1.0), (500.0, 1.0), (500.005, 1.0), (500.02, 1.0)]
        spec = make_spectrum(peaks)
        index = build_experimental_spectrum_index(spec, bin_width=0.5)

        tol_ppm = Tolerance(20.0, unit="ppm", min_tolerance=0.001)
        left, right = get_matching_range(index, 500.0, tol_ppm)
        assert [p.mass for p in index.sorted_peaks[left:right]] == [499.995, 500.0, 500.005]

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

    def test_indexed_vs_direct_matching_equivalence(self) -> None:
        """Verify that matching with a pre-built spec_index produces identical MatchResult."""
        peaks = [(100.01, 10.0), (200.02, 20.0), (300.03, 30.0), (400.04, 40.0)]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - PROTON_MASS, 200.0 - PROTON_MASS], structure_mass=500.0, monomer_count=2
        )

        res_direct = match_spectrum(
            spec, theo, tolerance=Tolerance(0.05), scoring=ScoringMode.INTENCITY
        )
        spec_idx = build_experimental_spectrum_index(spec)
        res_indexed = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.05),
            scoring=ScoringMode.INTENCITY,
            spec_index=spec_idx,
        )

        assert res_direct.matched_peaks_count == res_indexed.matched_peaks_count
        assert res_direct.score == pytest.approx(res_indexed.score)
        assert len(res_direct.matched_peaks) == len(res_indexed.matched_peaks)

    def test_parallel_vs_sequential_matching_equivalence(self, monkeypatch) -> None:
        """Verify that parallel matching produces identical results to sequential matching."""
        import os

        from kakapo.models import ModificationMode

        peaks_1 = [(100.01, 10.0), (200.02, 20.0), (300.03, 30.0)]
        peaks_2 = [(150.01, 15.0), (250.02, 25.0), (350.03, 35.0)]
        spectra = [
            make_spectrum(peaks_1, precursor_mass=500.0, spectrum_id=f"spec_{i}")
            if i % 2 == 0
            else make_spectrum(peaks_2, precursor_mass=600.0, spectrum_id=f"spec_{i}")
            for i in range(10)
        ]

        theo_spectra = [
            make_theoretical_spectrum(
                [100.0 - PROTON_MASS, 200.0 - PROTON_MASS],
                structure_mass=500.0,
                monomer_count=2,
                structure_id=f"struct_{i}",
            )
            for i in range(5)
        ]

        # Force sequential execution path by mocking cpu_count to 1
        monkeypatch.setattr(os, "cpu_count", lambda: 1)
        res_seq = match_spectra(
            spectra=spectra,
            theoretical_spectra=theo_spectra,
            tolerance=Tolerance(0.05),
            scoring=ScoringMode.INTENCITY,
            min_shared_peaks=1,
            mode=ModificationMode.FREE_MOD,
        )

        # Force parallel execution path by mocking cpu_count to 4
        monkeypatch.setattr(os, "cpu_count", lambda: 4)
        res_para = match_spectra(
            spectra=spectra,
            theoretical_spectra=theo_spectra,
            tolerance=Tolerance(0.05),
            scoring=ScoringMode.INTENCITY,
            min_shared_peaks=1,
            mode=ModificationMode.FREE_MOD,
        )

        assert len(res_seq) == len(res_para)
        res_seq_sorted = sorted(res_seq, key=lambda r: (r.spectrum_id, r.structure_id))
        res_para_sorted = sorted(res_para, key=lambda r: (r.spectrum_id, r.structure_id))

        for r_s, r_p in zip(res_seq_sorted, res_para_sorted):
            assert r_s.spectrum_id == r_p.spectrum_id
            assert r_s.structure_id == r_p.structure_id
            assert r_s.matched_peaks_count == r_p.matched_peaks_count
            assert r_s.score == pytest.approx(r_p.score)
            assert len(r_s.matched_peaks) == len(r_p.matched_peaks)
