"""Unit tests for kakapo data models and their validations."""

import numpy as np
import pytest

from kakapo.models import (
    Chirality,
    MatchedPeak,
    MatchResult,
    Molecule,
    Monomer,
    MonomerModification,
    MonomerOrigin,
    NerpaMSData,
    Spectrum,
    TheoreticalIonWithMass,
    Tolerance,
)

DUMMY_SPEC = Spectrum(
    spectrum_id="spec_1",
    source_file_path="test_file.mgf",
    mzs=np.array([100.0]),
    intensities=np.array([1.0]),
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
                mzs=np.array([]),
                intensities=np.array([]),
                precursor_mass=500.0,
            )

    def test_spectrum_negative_precursor_mass(self):
        """Spectrum cannot have negative precursor mass."""
        with pytest.raises(ValueError, match="non-negative"):
            Spectrum(
                spectrum_id="spec_1",
                source_file_path="test_file.mgf",
                mzs=np.array([100.0]),
                intensities=np.array([1.0]),
                precursor_mass=-1.0,
            )

    def test_spectrum_with_charge(self):
        """Spectrum stores and validates charge."""
        spec = Spectrum(
            spectrum_id="spec_1",
            source_file_path="test_file.mgf",
            mzs=np.array([100.0]),
            intensities=np.array([1.0]),
            precursor_mass=500.0,
            charge=2,
        )
        assert spec.charge == 2

        spec_zero = Spectrum(
            spectrum_id="spec_1",
            source_file_path="test_file.mgf",
            mzs=np.array([100.0]),
            intensities=np.array([1.0]),
            precursor_mass=500.0,
            charge=0,
        )
        assert spec_zero.charge is None

        with pytest.raises(TypeError, match="Charge must be an integer or None"):
            Spectrum(
                spectrum_id="spec_1",
                source_file_path="test_file.mgf",
                mzs=np.array([100.0]),
                intensities=np.array([1.0]),
                precursor_mass=500.0,
                charge="2",  # type: ignore
            )

    def test_spectrum_source_file_path_validation(self):
        """Spectrum validates that source_file_path is a string."""
        with pytest.raises(TypeError, match="Source file path must be a string"):
            Spectrum(
                spectrum_id="spec_1",
                source_file_path=123,  # type: ignore
                mzs=np.array([100.0]),
                intensities=np.array([1.0]),
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
                tolerance=Tolerance(0.01),
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
                tolerance=Tolerance(0.01),
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
                tolerance=Tolerance(0.01),
                matched_peaks_count=5,
                theoretical_peaks_count=10,
                experimental_peaks_count=12,
                monomer_count=3,
                unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
            )

    def test_match_result_invalid_tolerance_type(self):
        """MatchResult validates that tolerance is a Tolerance instance."""
        with pytest.raises(TypeError, match="tolerance must be a Tolerance instance"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.5,
                tolerance="not a tolerance instance",  # type: ignore
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
                theoretical_mz=-1.0,
                experimental_mz=100.0,
                experimental_peak_idx=0,
                charge=1,
                theoretical_fragment_mask=1,
            )

    def test_matched_peak_negative_experimental_mz(self):
        """MatchedPeak validates that experimental_mz is non-negative."""
        with pytest.raises(ValueError, match="Experimental m/z must be non-negative"):
            MatchedPeak(
                theoretical_mz=100.0,
                experimental_mz=-1.0,
                experimental_peak_idx=0,
                charge=1,
                theoretical_fragment_mask=1,
            )

    def test_matched_peak_zero_charge(self):
        """MatchedPeak validates that charge is positive (fails for zero)."""
        with pytest.raises(ValueError, match="Charge must be positive"):
            MatchedPeak(
                theoretical_mz=100.0,
                experimental_mz=100.0,
                experimental_peak_idx=0,
                charge=0,
                theoretical_fragment_mask=1,
            )

    def test_matched_peak_negative_charge(self):
        """MatchedPeak validates that charge is positive (fails for negative)."""
        with pytest.raises(ValueError, match="Charge must be positive"):
            MatchedPeak(
                theoretical_mz=100.0,
                experimental_mz=100.0,
                experimental_peak_idx=0,
                charge=-1,
                theoretical_fragment_mask=1,
            )

    def test_matched_peak_valid_initialization(self):
        """MatchedPeak initializes successfully with valid inputs."""
        ep = MatchedPeak(
            theoretical_mz=100.0,
            experimental_mz=100.1,
            experimental_peak_idx=5,
            charge=2,
            theoretical_fragment_mask=3,
        )
        assert ep.theoretical_mz == 100.0
        assert ep.experimental_mz == 100.1
        assert ep.experimental_peak_idx == 5
        assert ep.charge == 2
        assert ep.theoretical_fragment_mask == 3

    def test_matched_peak_invalid_mask_type(self):
        """MatchedPeak validates that theoretical_fragment_mask is an integer."""
        with pytest.raises(TypeError, match="Theoretical fragment mask must be an integer"):
            MatchedPeak(
                theoretical_mz=100.0,
                experimental_mz=100.0,
                experimental_peak_idx=0,
                charge=1,
                theoretical_fragment_mask="not an int",  # type: ignore
            )

    def test_matched_peak_negative_mask(self):
        """MatchedPeak validates that theoretical_fragment_mask is positive."""
        with pytest.raises(ValueError, match="Theoretical fragment mask must be positive"):
            MatchedPeak(
                theoretical_mz=100.0,
                experimental_mz=100.0,
                experimental_peak_idx=0,
                charge=1,
                theoretical_fragment_mask=-1,
            )

    def test_matched_peak_invalid_index_type(self):
        """MatchedPeak validates that experimental_peak_idx is an integer."""
        with pytest.raises(TypeError, match="Experimental peak index must be an integer"):
            MatchedPeak(
                theoretical_mz=100.0,
                experimental_mz=100.0,
                charge=1,
                theoretical_fragment_mask=1,
                experimental_peak_idx="not an int",  # type: ignore
            )

    def test_matched_peak_negative_index(self):
        """MatchedPeak validates that experimental_peak_idx is non-negative."""
        with pytest.raises(ValueError, match="Experimental peak index must be non-negative"):
            MatchedPeak(
                theoretical_mz=100.0,
                experimental_mz=100.0,
                charge=1,
                theoretical_fragment_mask=1,
                experimental_peak_idx=-1,
            )

    def test_match_result_default_matched_peaks(self):
        """MatchResult has an empty list of matched peaks by default."""
        mr = MatchResult(
            structure_id="mol_1",
            spectrum_id="spec_1",
            structure_mass=100.0,
            spectrum_mass=100.5,
            tolerance=Tolerance(0.01),
            matched_peaks_count=5,
            theoretical_peaks_count=10,
            experimental_peaks_count=15,
            monomer_count=3,
            unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
        )
        assert mr.matched_peaks == []
        assert isinstance(mr.result_id, str)
        assert len(mr.result_id) > 0
        assert mr.duplicate_representative is None
        assert mr.post_modification_ion_masses == []

    def test_match_result_result_id_validation(self):
        """MatchResult validates that result_id is a string."""
        base_args = {
            "structure_id": "mol_1",
            "spectrum_id": "spec_1",
            "structure_mass": 100.0,
            "spectrum_mass": 100.5,
            "tolerance": Tolerance(0.01),
            "matched_peaks_count": 5,
            "theoretical_peaks_count": 10,
            "experimental_peaks_count": 15,
            "monomer_count": 3,
            "unmodified_monomer_mass_vector": (100.0, 100.0, 100.0),
        }

        with pytest.raises(TypeError, match="result_id must be a string"):
            MatchResult(**(base_args | {"result_id": 123}))

    def test_match_result_duplicate_representative_validation(self):
        """MatchResult validates that duplicate_representative is a string or None."""
        base_args = {
            "structure_id": "mol_1",
            "spectrum_id": "spec_1",
            "structure_mass": 100.0,
            "spectrum_mass": 100.5,
            "tolerance": Tolerance(0.01),
            "matched_peaks_count": 5,
            "theoretical_peaks_count": 10,
            "experimental_peaks_count": 15,
            "monomer_count": 3,
            "unmodified_monomer_mass_vector": (100.0, 100.0, 100.0),
        }

        with pytest.raises(TypeError, match="duplicate_representative must be a string or None"):
            MatchResult(**(base_args | {"duplicate_representative": 123}))

    def test_match_result_post_modification_ion_masses_validation(self):
        """MatchResult validates that post_modification_ion_masses is a list."""
        base_args = {
            "structure_id": "mol_1",
            "spectrum_id": "spec_1",
            "structure_mass": 100.0,
            "spectrum_mass": 100.5,
            "tolerance": Tolerance(0.01),
            "matched_peaks_count": 5,
            "theoretical_peaks_count": 10,
            "experimental_peaks_count": 15,
            "monomer_count": 3,
            "unmodified_monomer_mass_vector": (100.0, 100.0, 100.0),
        }

        with pytest.raises(TypeError, match="post_modification_ion_masses must be a list"):
            MatchResult(**(base_args | {"post_modification_ion_masses": "not a list"}))

    def test_match_result_negative_experimental_peaks_count(self):
        """MatchResult validates that experimental peaks count is non-negative."""
        with pytest.raises(ValueError, match="Experimental peaks count must be non-negative"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.5,
                tolerance=Tolerance(0.01),
                matched_peaks_count=5,
                theoretical_peaks_count=10,
                experimental_peaks_count=-1,
                monomer_count=3,
                unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
            )

    def test_monomer_modification_validation(self):
        """MonomerModification validates fields correctly."""
        with pytest.raises(ValueError, match="monomer_idx must be a non-negative integer"):
            MonomerModification(
                monomer_idx=-1,
                initial_monomer_name="Ala",
                initial_monomer_mass=100.0,
                mass_difference=1.0,
            )

        with pytest.raises(ValueError, match="initial_monomer_name cannot be empty"):
            MonomerModification(
                monomer_idx=0,
                initial_monomer_name="",
                initial_monomer_mass=100.0,
                mass_difference=1.0,
            )

        with pytest.raises(ValueError, match="initial_monomer_mass must be non-negative"):
            MonomerModification(
                monomer_idx=0,
                initial_monomer_name="Ala",
                mass_difference=1.0,
                initial_monomer_mass=-5.0,
            )

    def test_match_result_invalid_modifications(self):
        """MatchResult validates modifications type."""
        with pytest.raises(TypeError, match="modifications must be a tuple"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.5,
                tolerance=Tolerance(0.01),
                matched_peaks_count=5,
                theoretical_peaks_count=10,
                experimental_peaks_count=15,
                monomer_count=3,
                unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
                modifications=[1, 2],  # type: ignore
            )

    def test_match_result_invalid_monomer_mass_vector_type(self):
        """MatchResult raises TypeError if unmodified_monomer_mass_vector is not a tuple."""
        with pytest.raises(TypeError, match="Unmodified monomer mass vector must be a tuple"):
            MatchResult(
                structure_id="mol_1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.5,
                tolerance=Tolerance(0.01),
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


class TestTolerance:
    """Unit tests for the Tolerance class."""

    def test_tolerance_creation_valid(self):
        t = Tolerance(value=20.0, unit="ppm", min_tolerance=0.005)
        assert t.value == 20.0
        assert t.unit == "ppm"
        assert t.min_tolerance == 0.005

    def test_tolerance_creation_negative_value(self):
        with pytest.raises(ValueError, match="Tolerance value must be non-negative"):
            Tolerance(value=-1.0)

    def test_tolerance_creation_invalid_unit(self):
        with pytest.raises(ValueError, match="Tolerance unit must be 'da' or 'ppm'"):
            Tolerance(value=1.0, unit="invalid")

    def test_tolerance_creation_negative_min_tolerance(self):
        with pytest.raises(ValueError, match="Minimum tolerance must be non-negative"):
            Tolerance(value=1.0, unit="ppm", min_tolerance=-0.1)

    def test_tolerance_from_string_ppm(self):
        t = Tolerance.from_string("20ppm")
        assert t.value == 20.0
        assert t.unit == "ppm"

    def test_tolerance_from_string_da(self):
        t = Tolerance.from_string("0.02da")
        assert t.value == 0.02
        assert t.unit == "da"

    def test_tolerance_from_string_default_unit(self):
        t = Tolerance.from_string("0.02")
        assert t.value == 0.02
        assert t.unit == "da"

    def test_tolerance_from_string_invalid_ppm(self):
        with pytest.raises(ValueError, match="Invalid ppm tolerance value"):
            Tolerance.from_string("abcppm")

    def test_tolerance_from_string_invalid_da(self):
        with pytest.raises(ValueError, match="Invalid Da tolerance value"):
            Tolerance.from_string("abcda")

    def test_tolerance_from_string_invalid_value(self):
        with pytest.raises(ValueError, match="Invalid tolerance value"):
            Tolerance.from_string("abc")

    def test_tolerance_is_within_da(self):
        t = Tolerance(value=0.02, unit="da")
        assert t.is_within(100.0, 100.01)
        assert not t.is_within(100.0, 100.03)

    def test_tolerance_is_within_ppm(self):
        t = Tolerance(value=20.0, unit="ppm", min_tolerance=0.0001)
        # At 100.0: 20 ppm = 0.002. Limit = 0.002.
        assert t.is_within(100.0, 100.0015)
        assert not t.is_within(100.0, 100.0025)

    def test_tolerance_is_within_ppm_min_limit(self):
        t = Tolerance(value=20.0, unit="ppm", min_tolerance=0.001)
        # At 10.0: 20 ppm = 0.0002. min_tolerance = 0.001. Limit = 0.001.
        assert t.is_within(10.0, 10.0005)
        assert not t.is_within(10.0, 10.0015)

    def test_tolerance_get_limit_da(self):
        t = Tolerance(value=0.02, unit="da")
        assert t.get_limit(100.0) == 0.02
        assert t.get_limit(1000.0) == 0.02

    def test_tolerance_get_limit_ppm(self):
        t = Tolerance(value=20.0, unit="ppm", min_tolerance=0.001)
        assert t.get_limit(100.0) == pytest.approx(0.002)
        assert t.get_limit(10.0) == pytest.approx(0.001)

    def test_tolerance_str(self):
        t = Tolerance(value=20.0, unit="ppm")
        assert str(t) == "20.0ppm"


class TestNerpaMSDataModels:
    """Granular unit tests for MonomerOrigin and NerpaMSData data models."""

    def test_monomer_origin_creation(self):
        """Test instantiation of MonomerOrigin."""
        origin = MonomerOrigin(compound_id="comp_1", monomer_name="Ala", monomer_idx=4)
        assert origin.compound_id == "comp_1"
        assert origin.monomer_name == "Ala"
        assert origin.monomer_idx == 4

    def test_monomer_origin_from_dict(self):
        """Test constructing MonomerOrigin from a raw dictionary."""
        raw = {"compound_id": "comp_2", "monomer_name": "Leu", "monomer_idx": 2}
        from_dict_origin = MonomerOrigin.from_dict(raw)
        assert from_dict_origin is not None
        assert from_dict_origin.compound_id == "comp_2"
        assert from_dict_origin.monomer_name == "Leu"
        assert from_dict_origin.monomer_idx == 2

    def test_monomer_origin_from_dict_none(self):
        """Test MonomerOrigin.from_dict with None or non-dict input."""
        assert MonomerOrigin.from_dict(None) is None
        assert MonomerOrigin.from_dict("invalid") is None  # type: ignore

    def test_nerpa_ms_data_creation(self):
        """Test instantiation of NerpaMSData."""
        origin = MonomerOrigin(compound_id="c1", monomer_name="Ala", monomer_idx=1)
        data = NerpaMSData(
            rank=2,
            score=12.5,
            old_to_new_mon_map=((None, 7), (1, 2)),
            monomer_origins=(origin,),
        )
        assert data.rank == 2
        assert data.score == 12.5
        assert data.old_to_new_mon_map == ((None, 7), (1, 2))
        assert data.monomer_origins == (origin,)

    def test_nerpa_ms_data_from_dict(self):
        """Test constructing NerpaMSData from a raw dictionary."""
        raw_dict = {
            "rank": 3,
            "score": 10.1,
            "old_to_new_mon_map": [[None, 5], [0, 1]],
            "monomer_origins": [{"compound_id": "c2", "monomer_name": "Val", "monomer_idx": 3}],
        }
        parsed = NerpaMSData.from_dict(raw_dict)
        assert parsed is not None
        assert parsed.rank == 3
        assert parsed.score == 10.1
        assert parsed.old_to_new_mon_map == ((None, 5), (0, 1))
        assert len(parsed.monomer_origins) == 1
        assert parsed.monomer_origins[0].compound_id == "c2"

    def test_nerpa_ms_data_from_dict_none(self):
        """Test NerpaMSData.from_dict with None or empty input."""
        assert NerpaMSData.from_dict(None) is None
        assert NerpaMSData.from_dict({}) is None

    def test_molecule_with_nerpa_ms_data(self):
        """Test creating a Molecule with nerpa_ms_data attached."""
        data = NerpaMSData(rank=1, score=5.0)
        mol = Molecule(
            structure_id="mol_nerpa",
            monomers={
                0: Monomer(
                    norine_name="Gly", residue="Gly", is_methylated=False, chirality=Chirality.L
                )
            },
            nerpa_ms_data=data,
        )
        assert mol.nerpa_ms_data is not None
        assert mol.nerpa_ms_data.rank == 1
        assert mol.nerpa_ms_data.score == 5.0
