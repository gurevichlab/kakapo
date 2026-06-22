import pytest

from kakapo.core.matcher import match_spectra, match_spectrum
from kakapo.models import Peak, Spectrum, TheoreticalIon, TheoreticalSpectrum


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
        TheoreticalIon(fragment_id=f"frag_{idx}", mass=mass, mask=(1 << idx))
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

    def test_exact_matches(self):
        """Test that exact matches are correctly identified and counted."""
        spec = make_spectrum([(100.0, 1.0), (200.0, 1.0), (300.0, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276, 200.0 - 1.007276, 300.0 - 1.007276],
            structure_mass=500.0,
            monomer_count=3,
        )

        result = match_spectrum(spec, theo, mass_tolerance=0.01)

        assert result.matched_peaks_count == 3
        assert result.theoretical_peaks_count == 3
        assert result.experimental_peaks_count == 3
        assert result.structure_id == "struct_1"
        assert result.spectrum_id == "spec_1"
        assert result.structure_mass == 500.0
        assert result.spectrum_mass == 500.0
        assert result.mass_tolerance == 0.01
        assert result.monomer_count == 3

    def test_matches_within_tolerance(self):
        """Test matching when peaks are within tolerance but not exact."""
        spec = make_spectrum([(100.01, 1.0), (200.02, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276, 200.0 - 1.007276], structure_mass=500.0, monomer_count=2
        )

        result = match_spectrum(spec, theo, mass_tolerance=0.03)
        assert result.matched_peaks_count == 2

        # Outside of tolerance
        result_fail = match_spectrum(spec, theo, mass_tolerance=0.005)
        assert result_fail.matched_peaks_count == 0

    def test_no_matches(self):
        """Test matching when there are no peaks in tolerance."""
        spec = make_spectrum([(150.0, 1.0), (250.0, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276, 200.0 - 1.007276],
            structure_mass=500.0,
            monomer_count=2,
        )

        result = match_spectrum(spec, theo, mass_tolerance=0.1)
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
        # theoretical ions:
        # 100.0 - PROTON_MASS => theoretical peak mz = 100.0 (matches 100.01 within 0.02)
        # 150.0 - PROTON_MASS => theoretical peak mz = 150.0 (no match)
        # 300.04 - PROTON_MASS => theoretical peak mz = 300.04 (matches 300.05 within 0.02)
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

    def test_matching_charge_out_of_range(self):
        """Test that peak matching is skipped when the matching charge exceeds spectrum charge."""
        # For charge 1, mz = ion.mass + 1.007276.
        # For charge 2, mz = (ion.mass + 2.014552) / 2.
        # Let's say ion.mass = 198.0.
        # Charge 1 mz = 199.007276.
        # Charge 2 mz = (198.0 + 2.014552) / 2 = 100.007276.
        theo = make_theoretical_spectrum([198.0], structure_mass=500.0, monomer_count=1)

        # Spectrum charge is 1: peak at 100.007276 (matches charge 2) should not match.
        spec = make_spectrum([(100.007276, 1.0)], precursor_mass=500.0, charge=1)
        result = match_spectrum(spec, theo, mass_tolerance=0.01)
        assert result.matched_peaks_count == 0

    def test_matching_charge_in_range(self):
        """Test that peaks matching a higher charge are identified when spectrum charge permits."""
        # For charge 2, mz = (ion.mass + 2.014552) / 2.
        # Let's say ion.mass = 198.0.
        # Charge 2 mz = (198.0 + 2.014552) / 2 = 100.007276.
        theo = make_theoretical_spectrum([198.0], structure_mass=500.0, monomer_count=1)

        # Spectrum charge is 2: peak at 100.007276 (matches charge 2) should match.
        spec = make_spectrum([(100.007276, 1.0)], precursor_mass=500.0, charge=2)
        result = match_spectrum(spec, theo, mass_tolerance=0.01)
        assert result.matched_peaks_count == 1
        assert result.matched_peaks[0].charge == 2
        assert result.matched_peaks[0].theoretical_mz == pytest.approx(100.007276)

    def test_matching_charge_default_fallback(self):
        """Test that spectrum charge defaults to MAX_CHARGE when not specified."""
        # For charge 2, mz = (ion.mass + 2.014552) / 2.
        # Let's say ion.mass = 198.0.
        # Charge 2 mz = (198.0 + 2.014552) / 2 = 100.007276.
        theo = make_theoretical_spectrum([198.0], structure_mass=500.0, monomer_count=1)

        # Spectrum charge is None (defaults to MAX_CHARGE = 2): peak matches at charge 2.
        spec = make_spectrum([(100.007276, 1.0)], precursor_mass=500.0, charge=None)
        result = match_spectrum(spec, theo, mass_tolerance=0.01)
        assert result.matched_peaks_count == 1
        assert result.matched_peaks[0].charge == 2

    def test_matching_multiple_charges(self):
        """Test matching peaks of different charges when spectrum charge allows it."""
        # ion 1 mass = 198.0 -> charge 1 mz = 199.007276, charge 2 mz = 100.007276
        # ion 2 mass = 298.0 -> charge 1 mz = 299.007276, charge 2 mz = 150.007276
        theo = make_theoretical_spectrum([198.0, 298.0], structure_mass=500.0, monomer_count=2)

        # Spectrum containing:
        # - peak at 199.007276 (matches ion 1 at charge 1)
        # - peak at 150.007276 (matches ion 2 at charge 2)
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
