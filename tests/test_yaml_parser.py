"""Unit tests for the YAML parser and its internal helpers."""

import os
import tempfile
from unittest.mock import patch

import networkx as nx
import pytest

from kakapo.io.bond_classifier import BondRule
from kakapo.io.yaml_parser import (
    _calculate_monomer_exact_mass,
    parse_rban_yaml,
    parse_rban_yaml_batch,
)
from kakapo.models import Molecule


class TestYamlParserBasic:
    """Test YAML parser with full Parsed_rBAN_Record structures."""

    def test_parse_parsed_rban_record_retains_atomic_payload(self):
        """Parse a full Parsed_rBAN_Record YAML and retain source atom graph."""
        yaml_data = """
compound_id: "NRP_PARSED_1"
metadata:
  name: "ParsedRecord"
  smiles: "NCC(=O)O"
  origin: null
  inchikey: null
  source: null
atoms:
  0:
    name: "N"
    hydrogens: 2
  1:
    name: "C"
    hydrogens: 1
  2:
    name: "C"
    hydrogens: 0
  3:
    name: "O"
    hydrogens: 0
  4:
    name: "O"
    hydrogens: 1
monomers:
  0:
    name: "Ala"
    atoms: [0, 1, 2, 3, 4]
    chirality: "L"
    is_pks_hybrid: false
monomer_bonds: []
atomic_bonds:
  - - [0, 1]
    - arity: "1"
      bond_type: "SINGLE"
  - - [1, 2]
    - arity: "1"
      bond_type: "SINGLE"
  - - [2, 3]
    - arity: "2"
      bond_type: "DOUBLE"
  - - [2, 4]
    - arity: "1"
      bond_type: "SINGLE"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            molecule = parse_rban_yaml(yaml_path)

            assert isinstance(molecule, Molecule)
            assert molecule.structure_id == "NRP_PARSED_1"
            assert molecule.smiles == "NCC(=O)O"
            assert len(molecule.monomers) == 1
            assert molecule.source_parsed_record is not None
            assert len(molecule.source_parsed_record.atomic_bonds) == 4

            # Verify monomer mass calculation check (alanine residue expected)
            ala = molecule.monomers[0]
            assert ala.mass is not None
            assert ala.mass > 70.0
        finally:
            os.unlink(yaml_path)

    def test_parse_parsed_rban_record_list(self):
        """Parse a YAML file containing a list of Parsed_rBAN_Record and return a dictionary of
        Molecules."""
        yaml_data = """
- compound_id: "NRP_PARSED_LIST_1"
  metadata:
    name: "ParsedRecord1"
    smiles: "NCC(=O)O"
    origin: null
    inchikey: null
    source: null
  atoms:
    0:
      name: "N"
      hydrogens: 2
  monomers:
    0:
      name: "Ala"
      atoms: [0]
      chirality: "L"
      is_pks_hybrid: false
  monomer_bonds: []
  atomic_bonds: []
- compound_id: "NRP_PARSED_LIST_2"
  metadata:
    name: "ParsedRecord2"
    smiles: "NCC(=O)O"
    origin: null
    inchikey: null
    source: null
  atoms:
    0:
      name: "N"
      hydrogens: 2
  monomers:
    0:
      name: "Ala"
      atoms: [0]
      chirality: "L"
      is_pks_hybrid: false
  monomer_bonds: []
  atomic_bonds: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            molecules = parse_rban_yaml(yaml_path)

            assert isinstance(molecules, dict)
            assert len(molecules) == 2
            assert "NRP_PARSED_LIST_1" in molecules
            assert "NRP_PARSED_LIST_2" in molecules
            assert isinstance(molecules["NRP_PARSED_LIST_1"], Molecule)
            assert isinstance(molecules["NRP_PARSED_LIST_2"], Molecule)
            assert molecules["NRP_PARSED_LIST_1"].structure_id == "NRP_PARSED_LIST_1"
            assert molecules["NRP_PARSED_LIST_2"].structure_id == "NRP_PARSED_LIST_2"
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
        """Should raise ValueError when Parsed_rBAN_Record fields are missing."""
        yaml_data = """
compound_id: "NRP005"
metadata:
  smiles: "NCC(=O)O"
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
        """Should raise ValueError when Parsed_rBAN_Record has no monomers."""
        yaml_data = """
compound_id: "NRP006"
metadata:
  name: "NoMonomers"
  smiles: "NCC(=O)O"
  origin: null
  inchikey: null
  source: null
atoms:
  0:
    name: "N"
    hydrogens: 2
atomic_bonds: []
monomer_bonds: []
monomers: {}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            with pytest.raises(ValueError, match="at least one monomer"):
                parse_rban_yaml(yaml_path)
        finally:
            os.unlink(yaml_path)


class TestYamlParserBatch:
    """Test batch parsing functionality."""

    def test_parse_batch_multiple_files(self):
        """Parse multiple YAML files at once."""
        yaml_data_1 = """
compound_id: "NRP_A"
metadata:
  name: null
  smiles: "NCC(=O)O"
  origin: null
  inchikey: null
  source: null
atoms:
  0:
    name: "N"
    hydrogens: 1
monomers:
  0:
    name: "Ala"
    atoms: [0]
    chirality: "L"
    is_pks_hybrid: false
monomer_bonds: []
atomic_bonds: []
"""
        yaml_data_2 = """
compound_id: "NRP_B"
metadata:
  name: null
  smiles: "NCC(=O)O"
  origin: null
  inchikey: null
  source: null
atoms:
  0:
    name: "N"
    hydrogens: 1
monomers:
  0:
    name: "Gly"
    atoms: [0]
    chirality: "L"
    is_pks_hybrid: false
monomer_bonds: []
atomic_bonds: []
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
compound_id: "NRP_Valid"
metadata:
  name: null
  smiles: "NCC(=O)O"
  origin: null
  inchikey: null
  source: null
atoms:
  0:
    name: "N"
    hydrogens: 1
monomers:
  0:
    name: "Ala"
    atoms: [0]
    chirality: "L"
    is_pks_hybrid: false
monomer_bonds: []
atomic_bonds: []
"""
        invalid_yaml = "this: is: not: valid:"

        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = os.path.join(tmpdir, "valid.yaml")
            invalid_path = os.path.join(tmpdir, "invalid.yaml")

            with open(valid_path, "w") as f:
                f.write(valid_yaml)
            with open(invalid_path, "w") as f:
                f.write(invalid_yaml)

            with pytest.raises(ValueError, match="Failed to parse") as excinfo:
                parse_rban_yaml_batch([valid_path, invalid_path])

            assert "invalid.yaml" in str(excinfo.value)


class MockMonomerInfo:
    """Mock for monomer info object used in tests."""

    def __init__(self, atoms):
        self.atoms = atoms


class MockAtomInfo:
    """Mock for atom info object used in tests."""

    def __init__(self, name, hydrogens):
        self.name = name
        self.hydrogens = hydrogens


class TestYamlParserMonomerMassCalculations:
    """Unit tests for _calculate_monomer_exact_mass helper."""

    def test_calculate_mass_valid(self):
        """Should correctly sum atomic masses with implicit hydrogens."""
        monomer = MockMonomerInfo(atoms=[0, 1])
        atoms = {
            0: MockAtomInfo(name="C", hydrogens=3),
            1: MockAtomInfo(name="O", hydrogens=1),
        }
        # C (12.0) + O (15.994915) + 4 * H (1.007825) = 32.026215
        expected_mass = 12.0 + 15.994915 + 4 * 1.007825
        mass = _calculate_monomer_exact_mass(monomer, atoms)
        assert mass == pytest.approx(expected_mass)

    def test_calculate_mass_unknown_element(self):
        """Should raise ValueError for unknown elements."""
        monomer = MockMonomerInfo(atoms=[0])
        atoms = {
            0: MockAtomInfo(name="Xx", hydrogens=0),
        }
        with pytest.raises(ValueError, match="Unknown element 'Xx'"):
            _calculate_monomer_exact_mass(monomer, atoms)

    def test_calculate_mass_missing_atom_id(self):
        """Should ignore missing atom IDs in the dictionary."""
        monomer = MockMonomerInfo(atoms=[0, 999])
        atoms = {
            0: MockAtomInfo(name="C", hydrogens=0),
        }
        # 999 is missing, should only calculate for atom 0
        mass = _calculate_monomer_exact_mass(monomer, atoms)
        assert mass == pytest.approx(12.0)

    def test_calculate_mass_empty_atoms(self):
        """Should return None if total mass is zero or no atoms are present."""
        monomer = MockMonomerInfo(atoms=[])
        assert _calculate_monomer_exact_mass(monomer, {}) is None


class TestYamlParserEdgeCases:
    """Additional edge cases for YAML parsing logic."""

    def test_invalid_chirality_mapping(self):
        """Invalid chirality name should raise KeyError during conversion."""
        yaml_data = """
compound_id: "NRP_CHIRAL_ERR"
metadata:
  name: null
  smiles: null
atoms:
  0:
    name: "N"
    hydrogens: 1
monomers:
  0:
    name: "Ala"
    atoms: [0]
    chirality: "INVALID_CHIRALITY"
    is_pks_hybrid: false
monomer_bonds: []
atomic_bonds: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            with pytest.raises(ValueError, match="Failed to parse Parsed_rBAN_Record"):
                parse_rban_yaml(yaml_path)
        finally:
            os.unlink(yaml_path)

    def test_mixed_list_errors(self):
        """Parsing lists with some invalid records should consolidate all errors."""
        yaml_data = """
- compound_id: "VALID_1"
  metadata: {name: null, smiles: null}
  atoms:
    0: {name: "N", hydrogens: 1}
  monomers:
    0: {name: "Ala", atoms: [0], chirality: "L", is_pks_hybrid: false}
  monomer_bonds: []
  atomic_bonds: []
- compound_id: "INVALID_1"
  metadata: {name: null, smiles: null}
  atoms: {}
  monomers: {}
  monomer_bonds: []
  atomic_bonds: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            with pytest.raises(
                ValueError, match="Failed to parse Parsed_rBAN_Record list"
            ) as excinfo:
                parse_rban_yaml(yaml_path)
            assert "Index 1:" in str(excinfo.value)
            assert "at least one monomer" in str(excinfo.value)
        finally:
            os.unlink(yaml_path)


class TestYamlParserBondClassificationIntegration:
    """Test bond classification integration with mocked rules."""

    def test_parse_rban_yaml_with_mocked_bond_rules(self):
        """Verify dynamic bond classification works using mocked BondRules."""
        yaml_data = """
compound_id: "NRP_BOND_TEST"
metadata:
  name: null
  smiles: "NCC"
atoms:
  0:
    name: "N"
    hydrogens: 1
  1:
    name: "C"
    hydrogens: 2
monomers:
  0:
    name: "M1"
    atoms: [0]
    chirality: "L"
  1:
    name: "M2"
    atoms: [1]
    chirality: "L"
monomer_bonds:
  - - [0, 1]
    - - monomer_to_atom: {0: 0, 1: 1}
        atomic_edge: {arity: "1", bond_type: "AMINO"}
atomic_bonds:
  - - [0, 1]
    - {arity: "1", bond_type: "AMINO"}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_data)
            yaml_path = f.name

        try:
            # Create a simple N-C mock rule graph
            mock_graph = nx.Graph()
            mock_graph.add_node(0, element="N")
            mock_graph.add_node(1, element="C")
            mock_graph.add_edge(0, 1)

            mock_rule = BondRule(
                name="amide_mock",
                mol_path="mock_path",
                mol_graph=mock_graph,
                mol_breakable_edge=(0, 1),
            )

            # Patch ALL_RULES in bond_classifier to use our mock rules
            with patch("kakapo.io.bond_classifier.ALL_RULES", [mock_rule]):
                molecule = parse_rban_yaml(yaml_path)
                assert isinstance(molecule, Molecule)
                assert len(molecule.bonds) == 1
                bond = molecule.bonds[0]
                assert len(bond.subtypes) == 1
                assert bond.subtypes[0].name == "amide_mock"
                # N (0) connects to C (1). 0 < 1. So reversed should be False.
                assert bond.subtypes[0].reversed is False
        finally:
            os.unlink(yaml_path)
