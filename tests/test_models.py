"""Unit tests for kakapo data models and their validations."""

import pytest

from kakapo.models import (
    MatchedPeak,
    MatchResult,
    Molecule,
    Peak,
    Spectrum,
    TheoreticalIonWithMass,
)

DUMMY_SPEC = Spectrum(
    spectrum_id="spec_1",
    source_file_path="test_file.mgf",
    peaks=frozenset([Peak(100.0, 1.0)]),
    precursor_mass=500.0,
)


class TestModelsValidation:
    """Test validation in kakapo models."""

    def test_spectrum_requires_peaks(self):
        """Spectrum cannot be created without peaks."""
        with pytest.raises(ValueError, match="at least one peak"):
            Spectrum(
                spectrum_id="spec_1",
                source_file_path="test_file.mgf",
                peaks=frozenset(),
                precursor_mass=500.0,
            )

    def test_spectrum_negative_precursor_mass(self):
        """Spectrum cannot have negative precursor mass."""
        with pytest.raises(ValueError, match="non-negative"):
            Spectrum(
                spectrum_id="spec_1",
                source_file_path="test_file.mgf",
                peaks=frozenset([Peak(100.0, 1.0)]),
                precursor_mass=-1.0,
            )

    def test_spectrum_with_charge(self):
        """Spectrum stores and validates charge."""
        spec = Spectrum(
            spectrum_id="spec_1",
            source_file_path="test_file.mgf",
            peaks=frozenset([Peak(100.0, 1.0)]),
            precursor_mass=500.0,
            charge=2,
        )
        assert spec.charge == 2

        spec_zero = Spectrum(
            spectrum_id="spec_1",
            source_file_path="test_file.mgf",
            peaks=frozenset([Peak(100.0, 1.0)]),
            precursor_mass=500.0,
            charge=0,
        )
        assert spec_zero.charge is None

        with pytest.raises(TypeError, match="Charge must be an integer or None"):
            Spectrum(
                spectrum_id="spec_1",
                source_file_path="test_file.mgf",
                peaks=frozenset([Peak(100.0, 1.0)]),
                precursor_mass=500.0,
                charge="2",  # type: ignore
            )

    def test_spectrum_source_file_path_validation(self):
        """Spectrum validates that source_file_path is a string."""
        with pytest.raises(TypeError, match="Source file path must be a string"):
            Spectrum(
                spectrum_id="spec_1",
                source_file_path=123,  # type: ignore
                peaks=frozenset([Peak(100.0, 1.0)]),
                precursor_mass=500.0,
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
                unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
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
                unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
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
                unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
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
                unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
            )

    def test_matched_peak_negative_theoretical_mz(self):
        """MatchedPeak validates that theoretical_mz is non-negative."""
        with pytest.raises(ValueError, match="Theoretical m/z must be non-negative"):
            MatchedPeak(
                theoretical_mz=-1.0, experimental_mz=100.0, charge=1, theoretical_fragment_mask=1
            )

    def test_matched_peak_negative_experimental_mz(self):
        """MatchedPeak validates that experimental_mz is non-negative."""
        with pytest.raises(ValueError, match="Experimental m/z must be non-negative"):
            MatchedPeak(
                theoretical_mz=100.0, experimental_mz=-1.0, charge=1, theoretical_fragment_mask=1
            )

    def test_matched_peak_zero_charge(self):
        """MatchedPeak validates that charge is positive (fails for zero)."""
        with pytest.raises(ValueError, match="Charge must be positive"):
            MatchedPeak(
                theoretical_mz=100.0, experimental_mz=100.0, charge=0, theoretical_fragment_mask=1
            )

    def test_matched_peak_negative_charge(self):
        """MatchedPeak validates that charge is positive (fails for negative)."""
        with pytest.raises(ValueError, match="Charge must be positive"):
            MatchedPeak(
                theoretical_mz=100.0, experimental_mz=100.0, charge=-1, theoretical_fragment_mask=1
            )

    def test_matched_peak_valid_initialization(self):
        """MatchedPeak initializes successfully with valid inputs."""
        ep = MatchedPeak(
            theoretical_mz=100.0, experimental_mz=100.1, charge=2, theoretical_fragment_mask=3
        )
        assert ep.theoretical_mz == 100.0
        assert ep.experimental_mz == 100.1
        assert ep.charge == 2
        assert ep.theoretical_fragment_mask == 3

    def test_matched_peak_invalid_mask_type(self):
        """MatchedPeak validates that theoretical_fragment_mask is an integer."""
        with pytest.raises(TypeError, match="Theoretical fragment mask must be an integer"):
            MatchedPeak(
                theoretical_mz=100.0,
                experimental_mz=100.0,
                charge=1,
                theoretical_fragment_mask="not an int",  # type: ignore
            )

    def test_matched_peak_negative_mask(self):
        """MatchedPeak validates that theoretical_fragment_mask is positive."""
        with pytest.raises(ValueError, match="Theoretical fragment mask must be positive"):
            MatchedPeak(
                theoretical_mz=100.0, experimental_mz=100.0, charge=1, theoretical_fragment_mask=-1
            )

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
            unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
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
                unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
            )

    def test_match_result_negative_initial_modified_monomer_mass(self):
        """MatchResult validates that initial modified monomer mass is non-negative."""
        with pytest.raises(ValueError, match="Initial modified monomer mass must be non-negative"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.5,
                mass_tolerance=0.01,
                matched_peaks_count=5,
                theoretical_peaks_count=10,
                experimental_peaks_count=15,
                monomer_count=3,
                unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
                initial_modified_monomer_mass=-1.0,
            )

    def test_match_result_invalid_monomer_mass_vector_type(self):
        """MatchResult raises TypeError if unmodified_monomer_mass_vector is not a tuple."""
        with pytest.raises(TypeError, match="Unmodified monomer mass vector must be a tuple"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.5,
                mass_tolerance=0.01,
                matched_peaks_count=5,
                theoretical_peaks_count=10,
                experimental_peaks_count=15,
                monomer_count=3,
                unmodified_monomer_mass_vector="not a tuple",  # type: ignore
            )

    def test_theoretical_ion_with_mass_invalid_inputs(self):
        """TheoreticalIonWithMass validates its fields."""
        with pytest.raises(ValueError, match="Fragment ID cannot be empty"):
            TheoreticalIonWithMass(fragment_id="", mass=100.0, mask=1)

        with pytest.raises(ValueError, match="Mask must be positive"):
            TheoreticalIonWithMass(fragment_id="frag_1", mass=100.0, mask=0)

        # Successful init
        ion = TheoreticalIonWithMass(fragment_id="frag_1", mass=100.0, mask=3)
        assert ion.fragment_id == "frag_1"
        assert ion.mass == 100.0
        assert ion.mask == 3
