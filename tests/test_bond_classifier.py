"""Tests for the dynamic bond classifier using SMARTS rules and RDKit."""

import pytest
from rdkit import Chem

from kakapo.io.bond_classifier import (
    BondRule,
    _build_local_mol,
    _match_rule,
    classify_single_bond,
    get_default_rules,
    make_rule,
)
from kakapo.io.json_parser import parse_rban_json
from kakapo.models import Atom, AtomicBond, AtomicEdge, Chirality, Molecule, Monomer


def test_redca_json_parsing_and_classification():
    """Verify that json_parser and bond_classifier successfully process REDCA records."""
    json_path = "/home/daria/Uni/Master/nerpa-ms-varquest/test_data/redca_parsed_rban_records.json"

    # This automatically runs parsing, mass precalculation, and bond classification!
    molecules = parse_rban_json(json_path)

    # 1. Basic Molecule Assertions
    assert isinstance(molecules, dict)
    molecule = molecules["RE001"]
    assert isinstance(molecule, Molecule)
    assert molecule.structure_id == "RE001"

    # Ala (Monomer at 0-indexed key 4) mass: formula C3H7NO2 - H2O = C3H5NO (monomer residue)
    # Atoms: [37, 35, 38, 34, 36] (C, C, O, N, C)
    # Expected mass: 71.037 Da
    ala = molecule.monomers[4]
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


def test_make_rule_success():
    """Verify make_rule compiles a valid SMARTS pattern with anchors."""
    rule = make_rule("test_amide", "[N:1](-[C:2]-O)-C")
    assert rule.name == "test_amide"
    assert rule.smarts == "[N:1](-[C:2]-O)-C"
    assert isinstance(rule.query_mol, Chem.Mol)
    # Mapping numbers 1 and 2 are at atom indices 0 and 1
    assert rule.new_u == 0
    assert rule.new_v == 1


def test_make_rule_invalid_smarts():
    """Verify make_rule raises ValueError for invalid SMARTS."""
    with pytest.raises(ValueError, match="Invalid SMARTS pattern"):
        make_rule("invalid", "[invalid_smarts_syntax")


def test_make_rule_missing_anchors():
    """Verify make_rule raises ValueError if anchors 1 or 2 are missing."""
    with pytest.raises(ValueError, match="does not contain map numbers 1 and 2"):
        make_rule("no_anchors", "N(-C-O)-C")


def test_get_default_rules():
    """Verify get_default_rules returns the compiled cached default rules."""
    rules = get_default_rules()
    assert len(rules) == 13
    assert all(isinstance(r, BondRule) for r in rules)
    # Check one of the rules
    amide_1 = next(r for r in rules if r.name == "amide_1")
    assert amide_1.smarts == "[N:1](-[C:2]-O)-C"


def test_build_local_mol():
    """Verify _build_local_mol constructs the correct RDKit Mol from monomer atoms."""
    monomer_a = Monomer(
        norine_name="M1",
        residue="R1",
        is_methylated=False,
        chirality=Chirality.UNKNOWN,
        atoms=(1, 2),
    )
    monomer_b = Monomer(
        norine_name="M2",
        residue="R2",
        is_methylated=False,
        chirality=Chirality.UNKNOWN,
        atoms=(3, 4),
    )

    atom_info = {
        1: Atom(name="C", hydrogens=3),
        2: Atom(name="O", hydrogens=0),
        3: Atom(name="N", hydrogens=1),
        4: Atom(name="C", hydrogens=3),
        5: Atom(name="S", hydrogens=0),  # Irrelevant
    }

    atomic_edges = {
        # Edge within monomer a
        AtomicEdge(1, 2): AtomicBond(arity="1", bond_type="SINGLE"),
        # Connecting edge between monomers
        AtomicEdge(2, 3): AtomicBond(arity="1", bond_type="SINGLE"),
        # Edge within monomer b
        AtomicEdge(3, 4): AtomicBond(arity="1", bond_type="SINGLE"),
        # Irrelevant edge outside the monomers
        AtomicEdge(4, 5): AtomicBond(arity="1", bond_type="SINGLE"),
    }

    local_mol, id_to_rdkit_idx = _build_local_mol(monomer_a, monomer_b, atom_info, atomic_edges)

    # Local mol should contain only atoms 1, 2, 3, 4
    assert local_mol.GetNumAtoms() == 4
    assert set(id_to_rdkit_idx.keys()) == {1, 2, 3, 4}

    # Verify atom symbols match
    assert local_mol.GetAtomWithIdx(id_to_rdkit_idx[1]).GetSymbol() == "C"
    assert local_mol.GetAtomWithIdx(id_to_rdkit_idx[2]).GetSymbol() == "O"
    assert local_mol.GetAtomWithIdx(id_to_rdkit_idx[3]).GetSymbol() == "N"
    assert local_mol.GetAtomWithIdx(id_to_rdkit_idx[4]).GetSymbol() == "C"

    # Verify single bonds are formed and non-aromatic
    for bond in local_mol.GetBonds():
        assert bond.GetBondType() == Chem.BondType.SINGLE
        assert not bond.GetIsAromatic()


def test_match_rule():
    """Verify _match_rule detects matches and their orientation correctly."""
    # Setup rule forward: N:1 - C:2
    rule_forward = make_rule("test_forward", "[N:1]-[C:2]")
    # Setup rule reversed: C:1 - N:2
    rule_reversed = make_rule("test_reversed", "[C:1]-[N:2]")

    # Target local mol: N (atom_id 10) - C (atom_id 11)
    # We build it using Chem.RWMol
    rw_mol = Chem.RWMol()
    n_idx = rw_mol.AddAtom(Chem.Atom("N"))
    c_idx = rw_mol.AddAtom(Chem.Atom("C"))
    rw_mol.AddBond(n_idx, c_idx, Chem.BondType.SINGLE)
    local_mol = rw_mol.GetMol()

    id_to_rdkit_idx = {10: n_idx, 11: c_idx}

    # Match rule forward
    # Edge is (10, 11) which connects N to C.
    # rule u (N) matches 10, rule v (C) matches 11.
    matched, is_reversed = _match_rule(local_mol, id_to_rdkit_idx, rule_forward, AtomicEdge(10, 11))
    assert matched is True
    assert is_reversed is False

    # Match rule reversed
    # Edge is (10, 11).
    # rule u (C) matches 11, rule v (N) matches 10.
    matched, is_reversed = _match_rule(
        local_mol, id_to_rdkit_idx, rule_reversed, AtomicEdge(10, 11)
    )
    assert matched is True
    assert is_reversed is True

    # Non-matching connection: (11, 12) connecting C to O (not matching N-C rule)
    rw_mol2 = Chem.RWMol()
    c2_idx = rw_mol2.AddAtom(Chem.Atom("C"))
    o_idx = rw_mol2.AddAtom(Chem.Atom("O"))
    rw_mol2.AddBond(c2_idx, o_idx, Chem.BondType.SINGLE)
    local_mol2 = rw_mol2.GetMol()

    id_to_rdkit_idx2 = {11: c2_idx, 12: o_idx}
    matched, is_reversed = _match_rule(
        local_mol2, id_to_rdkit_idx2, rule_forward, AtomicEdge(11, 12)
    )
    assert matched is False


@pytest.mark.parametrize(
    "atom_a_id, atom_b_id, is_reversed_match, expected_reversed",
    [
        (10, 20, False, False),  # Forward match, atom_a_id < atom_b_id -> False
        (20, 10, False, False),  # Forward match, atom_a_id > atom_b_id -> False
        (10, 20, True, True),  # Reversed match, atom_a_id < atom_b_id -> True
        (20, 10, True, True),  # Reversed match, atom_a_id > atom_b_id -> True
    ],
)
def test_classify_single_bond_directionality_xor(
    atom_a_id, atom_b_id, is_reversed_match, expected_reversed
):
    """Verify the XOR directionality logic in classify_single_bond."""
    # Setup rule: C:1 - N:2 or N:1 - C:2
    if not is_reversed_match:
        rule = make_rule("test_rule", "[C:1]-[N:2]")
    else:
        rule = make_rule("test_rule", "[N:1]-[C:2]")

    monomer_a = Monomer(
        norine_name="Ala",
        residue="Ala",
        is_methylated=False,
        chirality=Chirality.UNKNOWN,
        atoms=(atom_a_id,),
    )
    monomer_b = Monomer(
        norine_name="Gly",
        residue="Gly",
        is_methylated=False,
        chirality=Chirality.UNKNOWN,
        atoms=(atom_b_id,),
    )

    atom_info_dict = {
        atom_a_id: Atom(name="C", hydrogens=0),
        atom_b_id: Atom(name="N", hydrogens=0),
    }

    edge = AtomicEdge(atom_a_id, atom_b_id)
    atomic_edges = {edge: AtomicBond(arity="1", bond_type="SINGLE")}

    mon_to_atom = {
        0: atom_a_id,
        1: atom_b_id,
    }

    subtypes = classify_single_bond(
        source_monomer_idx=0,
        target_monomer_idx=1,
        monomer_a=monomer_a,
        monomer_b=monomer_b,
        mon_to_atom=mon_to_atom,
        atom_info_dict=atom_info_dict,
        atomic_edges=atomic_edges,
        rules=[rule],
    )

    assert len(subtypes) == 1
    assert subtypes[0].name == "test_rule"
    assert subtypes[0].reversed == expected_reversed
