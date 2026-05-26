"""Unit tests for input/output parsers."""

import os
import tempfile

import pytest
from monomer_names_helper import Chirality as NerpaChirality
from monomer_names_helper import NerpaResidue, NorineMonomerName
from rban_parsing.rban_monomer import rBAN_Monomer

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
                monomers=tuple(),
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
