"""Unit tests for the spectrum matching engine."""

from kakapo.core.matcher import match_spectrum
from kakapo.models import Peak, Spectrum, TheoreticalIon, TheoreticalSpectrum


def make_spectrum(peaks_data: list[tuple[float, float]], precursor_mass: float = 500.0) -> Spectrum:
    """Helper to create a Spectrum from list of mass-intensity tuples."""
    peaks = frozenset(Peak(mass=m, intensity=i) for m, i in peaks_data)
    return Spectrum(spectrum_id="spec_1", peaks=peaks, precursor_mass=precursor_mass)


def make_theoretical_spectrum(
    ion_masses: list[float], structure_mass: float = 500.0, monomer_count: int = 4
) -> TheoreticalSpectrum:
    """Helper to create a TheoreticalSpectrum from list of masses."""
    ions = frozenset(
        TheoreticalIon(fragment_id=f"frag_{idx}", mass=mass) for idx, mass in enumerate(ion_masses)
    )
    return TheoreticalSpectrum(
        structure_id="struct_1",
        ions=ions,
        structure_mass=structure_mass,
        monomer_count=monomer_count,
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
