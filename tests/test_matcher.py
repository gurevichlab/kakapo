import pytest

from kakapo.core.matcher import match_spectra, match_spectrum
from kakapo.models import Peak, RuleShift, Spectrum, TheoreticalIon, TheoreticalSpectrum


def make_spectrum(
    peaks_data: list[tuple[float, float]],
    precursor_mass: float = 500.0,
    charge: int | None = None,
    spectrum_id: str = "spec_1",
) -> Spectrum:
    """Helper to create a Spectrum from list of mass-intensity tuples."""
    peaks = frozenset(Peak(mass=m, intensity=i) for m, i in peaks_data)
    return Spectrum(
        spectrum_id=spectrum_id, peaks=peaks, precursor_mass=precursor_mass, charge=charge
    )


def make_theoretical_spectrum(
    ion_masses: list[float],
    structure_mass: float = 500.0,
    monomer_count: int = 4,
    structure_id: str = "struct_1",
) -> TheoreticalSpectrum:
    """Helper to create a TheoreticalSpectrum from list of masses."""
    ions = frozenset(
        TheoreticalIon(
            fragment_id=f"frag_{idx}",
            mask=(1 << idx),
            rules_shifts=(RuleShift("TEST", mass - 125.0),),
        )
        for idx, mass in enumerate(ion_masses)
    )
    return TheoreticalSpectrum(
        structure_id=structure_id,
        ions=ions,
        structure_mass=structure_mass,
        monomer_count=monomer_count,
        monomer_mass_vector=(125.0,) * monomer_count,
    )


class TestSpectrumMatcher:
    """Unit tests for the spectrum matching engine."""

    @pytest.mark.parametrize(
        "peaks,ion_masses,tolerance,expected_count",
        [
            # Exact matches
            (
                [(100.0, 1.0), (200.0, 1.0), (300.0, 1.0)],
                [100.0 - 1.007276, 200.0 - 1.007276, 300.0 - 1.007276],
                0.01,
                3,
            ),
            # Within tolerance
            ([(100.01, 1.0), (200.02, 1.0)], [100.0 - 1.007276, 200.0 - 1.007276], 0.03, 2),
            # Outside tolerance
            ([(100.01, 1.0), (200.02, 1.0)], [100.0 - 1.007276, 200.0 - 1.007276], 0.005, 0),
            # No matches
            ([(150.0, 1.0), (250.0, 1.0)], [100.0 - 1.007276, 200.0 - 1.007276], 0.1, 0),
        ],
    )
    def test_basic_matching_scenarios(self, peaks, ion_masses, tolerance, expected_count):
        """Test basic peak matching with exact, within tolerance, and no match scenarios."""
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(spec, theo, mass_tolerance=tolerance)
        assert result.matched_peaks_count == expected_count

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
        result1 = match_spectrum(spec, theo, mass_tolerance=0.05)
        assert result1.matched_peaks_count == 1

        # Two experimental peaks close to the same theoretical ion
        spec2 = make_spectrum([(99.99, 1.0), (100.01, 1.0)], precursor_mass=500.0)
        theo2 = make_theoretical_spectrum([100.0 - 1.007276], structure_mass=500.0, monomer_count=1)

        # Only 1 match should occur because there is only 1 theoretical ion
        result2 = match_spectrum(spec2, theo2, mass_tolerance=0.05)
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
        result = match_spectrum(spec, theo, mass_tolerance=0.02)
        assert result.matched_peaks_count == 2

    def test_matched_peaks_populated(self):
        """Verify matched_peaks details are populated in MatchResult."""
        spec = make_spectrum([(100.01, 1.0), (200.0, 1.0), (300.05, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276, 150.0 - 1.007276, 300.04 - 1.007276],
            structure_mass=500.0,
            monomer_count=3,
        )

        result = match_spectrum(spec, theo, mass_tolerance=0.02)
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
        result = match_spectrum(spec, theo, mass_tolerance=0.01)
        assert result.matched_peaks_count == expected_count
        if expected_count > 0:
            assert result.matched_peaks[0].charge == expected_charge

    def test_matching_multiple_charges(self):
        """Test matching peaks of different charges when spectrum charge allows it."""
        theo = make_theoretical_spectrum([198.0, 298.0], structure_mass=500.0, monomer_count=2)
        spec = make_spectrum([(199.007276, 1.0), (150.007276, 1.0)], precursor_mass=500.0, charge=2)
        result = match_spectrum(spec, theo, mass_tolerance=0.01)

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
            [100.0 - 1.007276], structure_mass=500.0, monomer_count=1, structure_id="struct_1"
        )
        theo2 = make_theoretical_spectrum(
            [200.0 - 1.007276], structure_mass=500.0, monomer_count=1, structure_id="struct_2"
        )

        results = match_spectra([spec1, spec2], [theo1, theo2], mass_tolerance=0.01)

        assert len(results) == 4

        # Key results by (spectrum_id, structure_id)
        results_map = {(r.spectrum_id, r.structure_id): r for r in results}

        assert results_map[("spec_1", "struct_1")].matched_peaks_count == 1
        assert results_map[("spec_1", "struct_2")].matched_peaks_count == 0
        assert results_map[("spec_2", "struct_1")].matched_peaks_count == 0
        assert results_map[("spec_2", "struct_2")].matched_peaks_count == 1

    def test_match_spectra_empty_inputs(self):
        """Test match_spectra with empty input iterables."""
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum([100.0 - 1.007276], structure_mass=500.0, monomer_count=1)

        assert len(match_spectra([], [theo], mass_tolerance=0.01)) == 0
        assert len(match_spectra([spec], [], mass_tolerance=0.01)) == 0
        assert len(match_spectra([], [], mass_tolerance=0.01)) == 0

    @pytest.mark.parametrize(
        "structure_mass,expected_match",
        [
            # Upper bound checks (precursor = 500.0, diff <= 300.0 -> max structure_mass = 800.0)
            (799.99, True),
            (800.0, True),
            (800.01, False),
            # Lower bound checks (precursor = 500.0, diff <= 300.0 -> min structure_mass = 200.0)
            (200.01, True),
            (200.0, True),
            (199.99, False),
        ],
    )
    def test_match_spectra_mass_difference_limits(self, structure_mass, expected_match):
        """Test pairwise matching filters pairs outside MAX_TOTAL_MASS_DIFFERENCE constraint."""
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_1")
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276], structure_mass=structure_mass, structure_id="struct_test"
        )
        results = match_spectra([spec], [theo], mass_tolerance=0.01)
        if expected_match:
            assert len(results) == 1
            assert results[0].structure_id == "struct_test"
        else:
            assert len(results) == 0

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

        theo = TheoreticalSpectrum(
            structure_id="struct_mod",
            ions=ions,
            structure_mass=sum(monomer_masses),
            monomer_count=len(monomer_masses),
            monomer_mass_vector=monomer_masses,
        )

        spec = make_spectrum(peaks, precursor_mass=precursor_mass, charge=1)
        result = match_spectrum(spec, theo, mass_tolerance=0.01)

        assert result.matched_peaks_count == expected_matched
        assert result.mass_mod == pytest.approx(expected_mod)
        assert result.modified_monomer_idx == expected_idx
