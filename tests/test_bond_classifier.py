"""Tests for the dynamic bond classifier and mass precalculation."""

import pytest

from kakapo.io.yaml_parser import parse_rban_yaml
from kakapo.models import Molecule


def test_redca_yaml_parsing_and_classification():
    """Verify that yaml_parser and bond_classifier successfully process REDCA records."""
    yaml_path = "/home/daria/Uni/Master/nerpa-ms-varquest/test_data/redca_parsed_rban_records.yaml"

    # This automatically runs parsing, mass precalculation, and bond classification!
    molecules = parse_rban_yaml(yaml_path)

    # 1. Basic Molecule Assertions
    assert isinstance(molecules, dict)
    molecule = molecules["RE001"]
    assert isinstance(molecule, Molecule)
    assert molecule.structure_id == "RE001"

    # 2. Monomer Mass Precalculation Assertions
    # Ala (Monomer 5) mass: formula C3H7NO2 - H2O = C3H5NO (monomer residue)
    # Atoms: [37, 35, 38, 34, 36] (C, C, O, N, C)
    # Expected mass: 71.037 Da
    ala = molecule.monomers[5]
    assert ala.norine_name == "Ala"
    assert ala.mass is not None
    assert pytest.approx(ala.mass, rel=1e-3) == 71.03718

    # 3. Bond Classification Assertions
    # All monomer bonds in REDCA are amide/peptide bonds.
    # So they should be classified as "amide_1"!
    assert len(molecule.bonds) > 0
    amide_bonds = [b for b in molecule.bonds if any(s.name == "amide_1" for s in b.subtypes)]
    assert len(amide_bonds) > 0

    for bond in molecule.bonds:
        # Check that subtypes list is populated
        assert len(bond.subtypes) > 0
        subtype_names = {s.name for s in bond.subtypes}
        assert any(name.startswith("amide_") for name in subtype_names)
        # Verify direction reversed is correctly boolean
        assert all(isinstance(s.reversed, bool) for s in bond.subtypes)
