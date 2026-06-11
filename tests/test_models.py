"""Unit tests for kakapo data models and their validations."""

import pytest

from kakapo.models import MatchedPeak, MatchResult, Molecule, Peak, Spectrum


class TestModelsValidation:
    """Test validation in kakapo models."""

    def test_spectrum_requires_peaks(self):
        """Spectrum cannot be created without peaks."""
        with pytest.raises(ValueError, match="at least one peak"):
            Spectrum(
                spectrum_id="spec_1",
                peaks=frozenset(),
                precursor_mass=500.0,
            )

    def test_spectrum_negative_precursor_mass(self):
        """Spectrum cannot have negative precursor mass."""
        with pytest.raises(ValueError, match="non-negative"):
            Spectrum(
                spectrum_id="spec_1",
                peaks=frozenset([Peak(100.0, 1.0)]),
                precursor_mass=-1.0,
            )

    def test_spectrum_with_charge(self):
        """Spectrum stores and validates charge."""
        spec = Spectrum(
            spectrum_id="spec_1",
            peaks=frozenset([Peak(100.0, 1.0)]),
            precursor_mass=500.0,
            charge=2,
        )
        assert spec.charge == 2

        with pytest.raises(TypeError, match="Charge must be an integer or None"):
            Spectrum(
                spectrum_id="spec_1",
                peaks=frozenset([Peak(100.0, 1.0)]),
                precursor_mass=500.0,
                charge="2",  # type: ignore
            )

    def test_molecule_requires_monomers(self):
        """Molecule cannot be created without monomers."""
        with pytest.raises(ValueError, match="at least one monomer"):
            Molecule(
                structure_id="mol_1",
                monomers={},
            )

    def test_match_result_requires_positive_counts(self):
        """MatchResult requires positive peak counts."""
        with pytest.raises(ValueError, match="positive"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.5,
                mass_tolerance=0.01,
                matched_peaks_count=5,
                theoretical_peaks_count=0,  # Invalid: must be > 0
                experimental_peaks_count=10,
                monomer_count=3,
            )

    def test_match_result_negative_structure_mass(self):
        """MatchResult validates that structure mass is non-negative."""
        with pytest.raises(ValueError, match="Structure mass must be non-negative"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=-1.0,
                spectrum_mass=100.5,
                mass_tolerance=0.01,
                matched_peaks_count=5,
                theoretical_peaks_count=10,
                experimental_peaks_count=12,
                monomer_count=3,
            )

    def test_match_result_negative_spectrum_mass(self):
        """MatchResult validates that spectrum mass is non-negative."""
        with pytest.raises(ValueError, match="Spectrum mass must be non-negative"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=-100.5,
                mass_tolerance=0.01,
                matched_peaks_count=5,
                theoretical_peaks_count=10,
                experimental_peaks_count=12,
                monomer_count=3,
            )

    def test_match_result_negative_mass_tolerance(self):
        """MatchResult validates that mass tolerance is non-negative."""
        with pytest.raises(ValueError, match="Mass tolerance must be non-negative"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.5,
                mass_tolerance=-0.01,
                matched_peaks_count=5,
                theoretical_peaks_count=10,
                experimental_peaks_count=12,
                monomer_count=3,
            )

    def test_matched_peak_negative_theoretical_mz(self):
        """MatchedPeak validates that theoretical_mz is non-negative."""
        with pytest.raises(ValueError, match="Theoretical m/z must be non-negative"):
            MatchedPeak(theoretical_mz=-1.0, experimental_mz=100.0, charge=1)

    def test_matched_peak_negative_experimental_mz(self):
        """MatchedPeak validates that experimental_mz is non-negative."""
        with pytest.raises(ValueError, match="Experimental m/z must be non-negative"):
            MatchedPeak(theoretical_mz=100.0, experimental_mz=-1.0, charge=1)

    def test_matched_peak_zero_charge(self):
        """MatchedPeak validates that charge is positive (fails for zero)."""
        with pytest.raises(ValueError, match="Charge must be positive"):
            MatchedPeak(theoretical_mz=100.0, experimental_mz=100.0, charge=0)

    def test_matched_peak_negative_charge(self):
        """MatchedPeak validates that charge is positive (fails for negative)."""
        with pytest.raises(ValueError, match="Charge must be positive"):
            MatchedPeak(theoretical_mz=100.0, experimental_mz=100.0, charge=-1)

    def test_matched_peak_valid_initialization(self):
        """MatchedPeak initializes successfully with valid inputs."""
        ep = MatchedPeak(theoretical_mz=100.0, experimental_mz=100.1, charge=2)
        assert ep.theoretical_mz == 100.0
        assert ep.experimental_mz == 100.1
        assert ep.charge == 2

    def test_match_result_default_matched_peaks(self):
        """MatchResult has an empty list of matched peaks by default."""
        mr = MatchResult(
            structure_id="mol_1",
            spectrum_id="spec_1",
            structure_mass=100.0,
            spectrum_mass=100.5,
            mass_tolerance=0.01,
            matched_peaks_count=5,
            theoretical_peaks_count=10,
            experimental_peaks_count=15,
            monomer_count=3,
        )
        assert mr.matched_peaks == []

    def test_match_result_negative_experimental_peaks_count(self):
        """MatchResult validates that experimental peaks count is non-negative."""
        with pytest.raises(ValueError, match="Experimental peaks count must be non-negative"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.5,
                mass_tolerance=0.01,
                matched_peaks_count=5,
                theoretical_peaks_count=10,
                experimental_peaks_count=-1,
                monomer_count=3,
            )
