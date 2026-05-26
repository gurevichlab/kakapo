"""Unit tests for input/output parsers."""

import os
import tempfile

import numpy as np
import pytest
from monomer_names_helper import Chirality as NerpaChirality
from monomer_names_helper import NerpaResidue, NorineMonomerName
from rban_parsing.rban_monomer import rBAN_Monomer

from kakapo.io.mgf_parser import (
    _get_precursor_mass,
    _get_spectrum_id,
    _matchms_to_kakapo,
    load_mgf,
    load_mgf_batch,
)
from kakapo.io.yaml_parser import (
    _convert_chirality,
    _rban_monomer_to_kakapo,
    parse_rban_yaml,
    parse_rban_yaml_batch,
)
from kakapo.models import Chirality, MatchResult, Molecule, Monomer, Peak, Spectrum


class TestChiralityConversion:
    """Test conversion from nerpa-ms to kakapo Chirality."""

    def test_convert_l_chirality(self):
        """L chirality should convert correctly."""
        result = _convert_chirality(NerpaChirality.L)
        assert result == Chirality.L

    def test_convert_d_chirality(self):
        """D chirality should convert correctly."""
        result = _convert_chirality(NerpaChirality.D)
        assert result == Chirality.D

    def test_convert_unknown_chirality(self):
        """Unknown chirality should convert correctly."""
        result = _convert_chirality(NerpaChirality.UNKNOWN)
        assert result == Chirality.UNKNOWN


class TestRbanMonomerConversion:
    """Test conversion from nerpa-ms rBAN_Monomer to kakapo Monomer."""

    def test_convert_basic_monomer(self):
        """Convert a basic rBAN_Monomer to kakapo Monomer."""
        rban_mon = rBAN_Monomer(
            residue=NerpaResidue("Ala"),
            methylated=False,
            chirality=NerpaChirality.L,
            is_pks_hybrid=False,
            rban_name=NorineMonomerName("Ala"),
            rban_idx=0,
        )

        result = _rban_monomer_to_kakapo(rban_mon)

        assert isinstance(result, Monomer)
        assert result.norine_name == "Ala"
        assert result.residue == "Ala"
        assert result.is_methylated is False
        assert result.chirality == Chirality.L

    def test_convert_methylated_monomer(self):
        """Convert a methylated monomer."""
        rban_mon = rBAN_Monomer(
            residue=NerpaResidue("Gly"),
            methylated=True,
            chirality=NerpaChirality.D,
            is_pks_hybrid=False,
            rban_name=NorineMonomerName("Gly"),
            rban_idx=1,
        )

        result = _rban_monomer_to_kakapo(rban_mon)

        assert result.is_methylated is True
        assert result.chirality == Chirality.D


class TestYamlParserBasic:
    """Test YAML parser with simple valid structures."""

    def test_parse_simple_linear_structure(self):
        """Parse a simple linear rBAN structure."""
        yaml_data = """
nrp_variant_id:
  nrp_id: "NRP001"
  variant_idx: 0
fragments:
  - is_cyclic: false
    monomers:
      - residue: "Ala"
        methylated: false
        chirality: "L"
        is_pks_hybrid: false
        rban_name: "Ala"
        rban_idx: 0
      - residue: "Gly"
        methylated: false
        chirality: "L"
        is_pks_hybrid: false
        rban_name: "Gly"
        rban_idx: 1
isolated_monomers: []
metadata:
  name: "TestPeptide"
  smiles: "C(C(=O)O)N"
  origin: null
  inchikey: null
  source: null
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            molecule = parse_rban_yaml(yaml_path)

            assert isinstance(molecule, Molecule)
            assert molecule.structure_id == "NRP001"
            assert len(molecule.monomers) == 2
            assert molecule.monomers[0].norine_name == "Ala"
            assert molecule.monomers[1].norine_name == "Gly"
            assert molecule.smiles == "C(C(=O)O)N"
        finally:
            os.unlink(yaml_path)

    def test_parse_cyclic_structure(self):
        """Parse a cyclic structure."""
        yaml_data = """
nrp_variant_id:
  nrp_id: "NRP002"
  variant_idx: 0
fragments:
  - is_cyclic: true
    monomers:
      - residue: "Pro"
        methylated: false
        chirality: "L"
        is_pks_hybrid: false
        rban_name: "Pro"
        rban_idx: 0
      - residue: "Leu"
        methylated: false
        chirality: "L"
        is_pks_hybrid: false
        rban_name: "Leu"
        rban_idx: 1
isolated_monomers: []
metadata:
  name: null
  smiles: null
  origin: null
  inchikey: null
  source: null
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            molecule = parse_rban_yaml(yaml_path)

            assert molecule.structure_id == "NRP002"
            assert len(molecule.monomers) == 2
        finally:
            os.unlink(yaml_path)

    def test_parse_with_methylation(self):
        """Parse structure with methylated monomers."""
        yaml_data = """
nrp_variant_id:
  nrp_id: "NRP003"
  variant_idx: 0
fragments:
  - is_cyclic: false
    monomers:
      - residue: "Asp"
        methylated: true
        chirality: "L"
        is_pks_hybrid: false
        rban_name: "Asp-Me"
        rban_idx: 0
isolated_monomers: []
metadata:
  name: null
  smiles: null
  origin: null
  inchikey: null
  source: null
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            molecule = parse_rban_yaml(yaml_path)

            assert molecule.monomers[0].is_methylated is True
        finally:
            os.unlink(yaml_path)

    def test_parse_with_pks_hybrid(self):
        """Parse structure with PKS hybrid monomers."""
        yaml_data = """
nrp_variant_id:
  nrp_id: "NRP004"
  variant_idx: 0
fragments:
  - is_cyclic: false
    monomers:
      - residue: "_PKS"
        methylated: false
        chirality: "UNKNOWN"
        is_pks_hybrid: true
        rban_name: "PKS-hybrid"
        rban_idx: 0
isolated_monomers: []
metadata:
  name: null
  smiles: null
  origin: null
  inchikey: null
  source: null
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            molecule = parse_rban_yaml(yaml_path)

            # The molecule should still be created even with PKS hybrid
            assert len(molecule.monomers) == 1
        finally:
            os.unlink(yaml_path)


class TestYamlParserErrors:
    """Test error handling in YAML parser."""

    def test_parse_nonexistent_file(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            parse_rban_yaml("/nonexistent/path/file.yaml")

    def test_parse_invalid_yaml(self):
        """Should raise ValueError for invalid YAML syntax."""
        invalid_yaml = """
this: is: invalid: yaml: syntax:
  - unbalanced brackets
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            yaml_path = f.name

        try:
            with pytest.raises(ValueError):
                parse_rban_yaml(yaml_path)
        finally:
            os.unlink(yaml_path)

    def test_parse_empty_yaml(self):
        """Should raise ValueError for empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            yaml_path = f.name

        try:
            with pytest.raises(ValueError, match="empty"):
                parse_rban_yaml(yaml_path)
        finally:
            os.unlink(yaml_path)

    def test_parse_missing_required_fields(self):
        """Should raise ValueError when required fields are missing."""
        yaml_data = """
nrp_variant_id:
  nrp_id: "NRP005"
  variant_idx: 0
fragments: []
isolated_monomers: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            with pytest.raises(ValueError):
                parse_rban_yaml(yaml_path)
        finally:
            os.unlink(yaml_path)

    def test_parse_missing_monomers(self):
        """Should raise ValueError when no monomers are present."""
        yaml_data = """
nrp_variant_id:
  nrp_id: "NRP006"
  variant_idx: 0
fragments:
  - is_cyclic: false
    monomers: []
isolated_monomers: []
metadata:
  name: null
  smiles: null
  origin: null
  inchikey: null
  source: null
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            with pytest.raises(ValueError, match="No monomers"):
                parse_rban_yaml(yaml_path)
        finally:
            os.unlink(yaml_path)


class TestYamlParserBatch:
    """Test batch parsing functionality."""

    def test_parse_batch_multiple_files(self):
        """Parse multiple YAML files at once."""
        yaml_data_1 = """
nrp_variant_id:
  nrp_id: "NRP_A"
  variant_idx: 0
fragments:
  - is_cyclic: false
    monomers:
      - residue: "Ala"
        methylated: false
        chirality: "L"
        is_pks_hybrid: false
        rban_name: "Ala"
        rban_idx: 0
isolated_monomers: []
metadata:
  name: null
  smiles: null
  origin: null
  inchikey: null
  source: null
"""
        yaml_data_2 = """
nrp_variant_id:
  nrp_id: "NRP_B"
  variant_idx: 0
fragments:
  - is_cyclic: false
    monomers:
      - residue: "Gly"
        methylated: false
        chirality: "L"
        is_pks_hybrid: false
        rban_name: "Gly"
        rban_idx: 0
isolated_monomers: []
metadata:
  name: null
  smiles: null
  origin: null
  inchikey: null
  source: null
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path_1 = os.path.join(tmpdir, "nrp_a.yaml")
            yaml_path_2 = os.path.join(tmpdir, "nrp_b.yaml")

            with open(yaml_path_1, "w") as f:
                f.write(yaml_data_1)
            with open(yaml_path_2, "w") as f:
                f.write(yaml_data_2)

            molecules = parse_rban_yaml_batch([yaml_path_1, yaml_path_2])

            assert len(molecules) == 2
            assert "NRP_A" in molecules
            assert "NRP_B" in molecules
            assert molecules["NRP_A"].structure_id == "NRP_A"
            assert molecules["NRP_B"].structure_id == "NRP_B"

    def test_parse_batch_with_error(self):
        """Batch parsing should report all errors."""
        valid_yaml = """
nrp_variant_id:
  nrp_id: "NRP_Valid"
  variant_idx: 0
fragments:
  - is_cyclic: false
    monomers:
      - residue: "Ala"
        methylated: false
        chirality: "L"
        is_pks_hybrid: false
        rban_name: "Ala"
        rban_idx: 0
isolated_monomers: []
metadata:
  name: null
  smiles: null
  origin: null
  inchikey: null
  source: null
"""
        invalid_yaml = "this: is: not: valid:"

        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = os.path.join(tmpdir, "valid.yaml")
            invalid_path = os.path.join(tmpdir, "invalid.yaml")

            with open(valid_path, "w") as f:
                f.write(valid_yaml)
            with open(invalid_path, "w") as f:
                f.write(invalid_yaml)

            with pytest.raises(ValueError, match="Failed to parse"):
                parse_rban_yaml_batch([valid_path, invalid_path])


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

    def test_molecule_requires_monomers(self):
        """Molecule cannot be created without monomers."""
        with pytest.raises(ValueError, match="at least one monomer"):
            Molecule(
                structure_id="mol_1",
                monomers=(),
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


class TestMgfParserHelpers:
    """Test helper functions for MGF parsing."""

    def test_get_precursor_mass_from_precursor_mz(self):
        """Extract precursor mass from precursor_mz field."""
        metadata = {
            "precursor_mz": 500.5,
            "spectrum_id": "spec_1",
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 500.5

    def test_get_precursor_mass_from_pepmass_float(self):
        """Extract precursor mass from pepmass as float."""
        metadata = {
            "pepmass": 600.0,
            "spectrum_id": "spec_2",
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 600.0

    def test_get_precursor_mass_from_pepmass_tuple(self):
        """Extract precursor mass from pepmass as tuple [mass, intensity]."""
        metadata = {
            "pepmass": (700.5, 1000.0),
            "spectrum_id": "spec_3",
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 700.5

    def test_get_precursor_mass_from_pepmass_list(self):
        """Extract precursor mass from pepmass as list."""
        metadata = {
            "pepmass": [750.0, 500.0],
            "spectrum_id": "spec_4",
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 750.0

    def test_get_precursor_mass_priority_precursor_mz(self):
        """precursor_mz should be prioritized over pepmass."""
        metadata = {
            "precursor_mz": 400.0,
            "pepmass": 500.0,
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 400.0

    def test_get_precursor_mass_missing_field(self):
        """Should raise ValueError when no precursor mass field is present."""
        metadata = {
            "spectrum_id": "spec_5",
        }
        with pytest.raises(ValueError, match="Could not extract"):
            _get_precursor_mass(metadata)

    def test_get_precursor_mass_negative_value(self):
        """Should raise ValueError for negative precursor mass."""
        metadata = {
            "precursor_mz": -100.0,
        }
        with pytest.raises(ValueError, match="Could not extract"):
            _get_precursor_mass(metadata)

    def test_get_spectrum_id_from_spectrum_id_field(self):
        """Extract spectrum_id from spectrum_id field."""
        metadata = {"spectrum_id": "spec_001"}
        spectrum_id = _get_spectrum_id(metadata)
        assert spectrum_id == "spec_001"

    def test_get_spectrum_id_from_scan_field(self):
        """Extract spectrum_id from scan field."""
        metadata = {"scan": "123"}
        spectrum_id = _get_spectrum_id(metadata)
        assert spectrum_id == "123"

    def test_get_spectrum_id_from_title_field(self):
        """Extract spectrum_id from title field."""
        metadata = {"title": "compound_A_ms2"}
        spectrum_id = _get_spectrum_id(metadata)
        assert spectrum_id == "compound_A_ms2"

    def test_get_spectrum_id_fallback(self):
        """Fall back to 'unknown' when no ID field is present."""
        metadata = {"precursor_mz": 500.0}
        spectrum_id = _get_spectrum_id(metadata)
        assert spectrum_id == "unknown"


class TestMgfParserConversion:
    """Test conversion from matchms to kakapo Spectrum."""

    def test_matchms_to_kakapo_basic_conversion(self):
        """Convert a basic matchms spectrum to kakapo Spectrum."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[100.0, 10.0], [200.0, 20.0], [300.0, 15.0]])
        metadata = {
            "spectrum_id": "spec_001",
            "precursor_mz": 400.0,
        }

        matchms_spec = MatchmsSpectrum(mz=peaks[:, 0], intensities=peaks[:, 1], metadata=metadata)
        kakapo_spec = _matchms_to_kakapo(matchms_spec)

        assert isinstance(kakapo_spec, Spectrum)
        assert kakapo_spec.spectrum_id == "spec_001"
        assert kakapo_spec.precursor_mass == 400.0
        assert len(kakapo_spec.peaks) == 3

    def test_matchms_to_kakapo_with_pepmass(self):
        """Convert spectrum using pepmass field."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[50.0, 5.0], [100.0, 10.0]])
        metadata = {
            "scan": "42",
            "pepmass": (250.5, 500.0),
        }

        matchms_spec = MatchmsSpectrum(mz=peaks[:, 0], intensities=peaks[:, 1], metadata=metadata)
        kakapo_spec = _matchms_to_kakapo(matchms_spec)

        assert kakapo_spec.spectrum_id == "42"
        assert kakapo_spec.precursor_mass == 250.5

    def test_matchms_to_kakapo_no_peaks(self):
        """Should raise ValueError for spectrum with no peaks."""
        from matchms import Spectrum as MatchmsSpectrum

        metadata = {"spectrum_id": "spec_empty", "precursor_mz": 300.0}

        matchms_spec = MatchmsSpectrum(mz=np.array([]), intensities=np.array([]), metadata=metadata)

        with pytest.raises(ValueError, match="has no peaks"):
            _matchms_to_kakapo(matchms_spec)

    def test_matchms_to_kakapo_invalid_peaks(self):
        """Should raise ValueError for invalid peak data."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[-10.0, 5.0], [100.0, 10.0]])
        metadata = {"spectrum_id": "spec_bad", "precursor_mz": 300.0}

        matchms_spec = MatchmsSpectrum(mz=peaks[:, 0], intensities=peaks[:, 1], metadata=metadata)

        with pytest.raises(ValueError, match="Invalid peak mass"):
            _matchms_to_kakapo(matchms_spec)


class TestMgfParserLoadFile:
    """Test loading MGF files."""

    def _create_test_mgf(self, content: str, filename: str = "test.mgf") -> str:
        """Helper to create a temporary MGF file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mgf", delete=False, prefix=filename.replace(".mgf", "")
        ) as f:
            f.write(content)
            return f.name

    def test_load_mgf_basic_file(self):
        """Load a basic MGF file with one spectrum."""
        mgf_content = """BEGIN IONS
TITLE=spectrum_1
PEPMASS=500.0
100 10
200 20
300 15
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            spectra = load_mgf(mgf_path, apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spectrum_1"
            assert spectra[0].precursor_mass == 500.0
            assert len(spectra[0].peaks) == 3
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_multiple_spectra(self):
        """Load MGF file with multiple spectra."""
        mgf_content = """BEGIN IONS
TITLE=spec_1
PEPMASS=500.0
100 10
200 20
END IONS
BEGIN IONS
TITLE=spec_2
PEPMASS=600.0
150 15
250 25
350 35
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            spectra = load_mgf(mgf_path, apply_filters=False)
            assert len(spectra) == 2
            assert spectra[0].spectrum_id == "spec_1"
            assert spectra[1].spectrum_id == "spec_2"
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_with_precursor_mz_field(self):
        """Load MGF using PRECURSORMZ field (common format variant)."""
        mgf_content = """BEGIN IONS
TITLE=spec_mz
PRECURSORMZ=550.5
100 10
200 20
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            spectra = load_mgf(mgf_path, apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].precursor_mass == 550.5
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_mgf("/nonexistent/path/file.mgf")

    def test_load_mgf_invalid_extension(self):
        """Should raise ValueError for non-.mgf files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="Expected .mgf"):
                load_mgf(txt_path)
        finally:
            os.unlink(txt_path)

    def test_load_mgf_empty_file(self):
        """Should raise ValueError for empty MGF file."""
        mgf_content = ""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            with pytest.raises(ValueError, match="No spectra"):
                load_mgf(mgf_path, apply_filters=False)
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_min_peaks_filter(self):
        """Should filter out spectra with too few peaks."""
        mgf_content = """BEGIN IONS
TITLE=few_peaks
PEPMASS=500.0
100 10
END IONS
BEGIN IONS
TITLE=many_peaks
PEPMASS=600.0
100 10
200 20
300 30
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            spectra = load_mgf(mgf_path, apply_filters=False, min_peaks=2)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "many_peaks"
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_batch_multiple_files(self):
        """Load multiple MGF files in batch."""
        mgf_content_1 = """BEGIN IONS
TITLE=spec_file1
PEPMASS=500.0
100 10
200 20
END IONS
"""
        mgf_content_2 = """BEGIN IONS
TITLE=spec_file2
PEPMASS=600.0
150 15
250 25
END IONS
"""
        mgf_path_1 = self._create_test_mgf(mgf_content_1, "file1.mgf")
        mgf_path_2 = self._create_test_mgf(mgf_content_2, "file2.mgf")

        try:
            all_spectra = load_mgf_batch([mgf_path_1, mgf_path_2], apply_filters=False)
            assert len(all_spectra) == 2
            assert all_spectra[0].spectrum_id == "spec_file1"
            assert all_spectra[1].spectrum_id == "spec_file2"
        finally:
            os.unlink(mgf_path_1)
            os.unlink(mgf_path_2)

    def test_load_mgf_batch_with_error(self):
        """Batch loading should propagate errors."""
        mgf_content_valid = """BEGIN IONS
TITLE=valid
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path_valid = self._create_test_mgf(mgf_content_valid, "valid.mgf")

        try:
            with pytest.raises(FileNotFoundError):
                load_mgf_batch([mgf_path_valid, "/nonexistent/file.mgf"], apply_filters=False)
        finally:
            os.unlink(mgf_path_valid)

    def test_load_mgf_with_paths_object(self):
        """load_mgf should accept both str and Path objects."""
        from pathlib import Path

        mgf_content = """BEGIN IONS
TITLE=spec_path
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            # Test with Path object
            spectra = load_mgf(Path(mgf_path), apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_path"

            # Test with string
            spectra = load_mgf(mgf_path, apply_filters=False)
            assert len(spectra) == 1
        finally:
            os.unlink(mgf_path)
