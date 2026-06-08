"""Unit tests for kakapo data models and their validations."""

import pytest

from kakapo.models import MatchResult, Molecule, Peak, Spectrum


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
                molecule_id="mol_1",
                spectrum_id="spec_1",
                score=0.8,
                matched_peaks_count=5,
                theoretical_peaks_count=0,  # Invalid: must be > 0
                monomer_count=3,
            )
