"""Tests for the dynamic bond classifier and mass precalculation."""

import logging

import networkx as nx
import pytest
from rdkit import Chem

from kakapo.io.bond_classifier import (
    BondRule,
    _build_local_graph,
    _match_rule,
    classify_single_bond,
    load_rules,
)
from kakapo.io.yaml_parser import parse_rban_yaml
from kakapo.models import Atom, AtomicBond, AtomicEdge, Chirality, Molecule, Monomer


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


def test_load_rules_non_existent_dir(caplog):
    """Verify load_rules warns and returns empty list for non-existent path."""
    with caplog.at_level(logging.WARNING):
        rules = load_rules("/non/existent/path")
    assert len(rules) == 0
    assert "Rules directory /non/existent/path does not exist." in caplog.text


def test_load_rules_invalid_mol_file(tmp_path, caplog):
    """Verify load_rules logs debug message and continues when MOL file cannot be parsed."""
    mol_path = tmp_path / "corrupt_rule.mol"
    # Write invalid content
    mol_path.write_text("INVALID_MOL_CONTENT")

    with caplog.at_level(logging.DEBUG):
        rules = load_rules(str(tmp_path))

    assert len(rules) == 0
    assert f"Failed to parse MOL file {mol_path}" in caplog.text


def test_load_rules_out_of_range_bond(tmp_path, caplog):
    """Verify load_rules handles out-of-range breakable bond index and logs warning."""
    # We will write a valid molecule with only 1 atom (hence 0 bonds)
    # but map it to a name in BREAKABLE_BOND_INDICES like "amide_1" (index 0)
    mol_path = tmp_path / "amide_1.mol"
    mol = Chem.MolFromSmiles("[C]")
    Chem.MolToMolFile(mol, str(mol_path))

    with caplog.at_level(logging.WARNING):
        rules = load_rules(str(tmp_path))

    assert len(rules) == 0
    assert "Bond index 0 out of range in MOL amide_1.mol" in caplog.text


def test_load_rules_success(tmp_path):
    """Verify load_rules successfully loads and constructs a BondRule."""
    mol_path = tmp_path / "amide_1.mol"
    # Create simple ethylamine: C-C-N (2 bonds: C-C index 0, C-N index 1)
    mol = Chem.MolFromSmiles("CCN")
    Chem.MolToMolFile(mol, str(mol_path))

    rules = load_rules(str(tmp_path))
    assert len(rules) == 1
    rule = rules[0]
    assert rule.name == "amide_1"
    assert rule.mol_path == str(mol_path)
    # The rule's breakable edge should be the bond at index 0 (connecting C to C)
    bond = mol.GetBondWithIdx(0)
    expected_edge = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    assert rule.mol_breakable_edge == expected_edge
    # Verify the constructed graph contains only non-H atoms (3 nodes: C, C, N)
    assert len(rule.mol_graph.nodes) == 3
    assert len(rule.mol_graph.edges) == 2


def test_build_local_graph():
    """Verify _build_local_graph constructs the correct subgraph from monomer atoms."""
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
        # Irrelevant atom that shouldn't be in the graph
        5: Atom(name="S", hydrogens=0),
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

    local_graph = _build_local_graph(monomer_a, monomer_b, atom_info, atomic_edges)

    # Local graph should contain nodes 1, 2, 3, 4 with correct attributes
    assert set(local_graph.nodes) == {1, 2, 3, 4}
    assert local_graph.nodes[1]["element"] == "C"
    assert local_graph.nodes[2]["element"] == "O"
    assert local_graph.nodes[3]["element"] == "N"
    assert local_graph.nodes[4]["element"] == "C"

    # Local graph should contain edges connecting them: (1, 2), (2, 3), (3, 4)
    # But not (4, 5) since 5 is not in monomer_a or monomer_b
    assert local_graph.has_edge(1, 2)
    assert local_graph.has_edge(2, 3)
    assert local_graph.has_edge(3, 4)
    assert not local_graph.has_edge(4, 5)


def test_match_rule():
    """Verify _match_rule detects matches and their orientation correctly."""
    # 1. Setup rule graph representing: C-O-N
    # Nodes: 0 (C), 1 (O), 2 (N)
    rule_graph = nx.Graph()
    rule_graph.add_node(0, element="C")
    rule_graph.add_node(1, element="O")
    rule_graph.add_node(2, element="N")
    rule_graph.add_edge(0, 1)
    rule_graph.add_edge(1, 2)

    rule_forward = BondRule(
        name="test_rule_forward",
        mol_path="mock_path",
        mol_graph=rule_graph,
        mol_breakable_edge=(1, 2),  # Breakable bond is O-N (1 to 2)
    )

    rule_reversed = BondRule(
        name="test_rule_reversed",
        mol_path="mock_path",
        mol_graph=rule_graph,
        mol_breakable_edge=(2, 1),  # Breakable bond is N-O (2 to 1)
    )

    # 2. Setup local graph matching rule: C-O-N (nodes 10, 11, 12)
    local_graph = nx.Graph()
    local_graph.add_node(10, element="C")
    local_graph.add_node(11, element="O")
    local_graph.add_node(12, element="N")
    local_graph.add_edge(10, 11)
    local_graph.add_edge(11, 12)

    # Edge (11, 12) canonicalized to (11, 12)
    # Forward rule: O(1)->N(2) maps 1->11, 2->12.
    # mapped_u=11 (atom_a), mapped_v=12 (atom_b).
    # Matches forward relative to ID order.
    matched, is_reversed = _match_rule(local_graph, rule_forward, AtomicEdge(11, 12))
    assert matched is True
    assert is_reversed is False

    # Reversed rule: N(2)->O(1) maps 2->12, 1->11.
    # mapped_u=12 (atom_b), mapped_v=11 (atom_a).
    # Matches reversed relative to ID order.
    matched, is_reversed = _match_rule(local_graph, rule_reversed, AtomicEdge(11, 12))
    assert matched is True
    assert is_reversed is True

    # Connecting edge 10-11 (C-O) is not the rule's breakable edge (O-N)
    matched, is_reversed = _match_rule(local_graph, rule_forward, AtomicEdge(10, 11))
    assert matched is False
    assert is_reversed is False


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
    # Setup rule graph representing: A -> B
    rule_graph = nx.Graph()
    rule_graph.add_node(0, element="C")
    rule_graph.add_node(1, element="N")
    rule_graph.add_edge(0, 1)

    rule = BondRule(
        name="test_rule",
        mol_path="mock_path",
        mol_graph=rule_graph,
        mol_breakable_edge=(0, 1) if not is_reversed_match else (1, 0),
    )

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


def test_classify_single_bond_defaults():
    """Verify classify_single_bond falls back to ALL_RULES when rules is None."""
    # Setup minimal monomers and atoms
    monomer_a = Monomer(
        norine_name="Ala",
        residue="Ala",
        is_methylated=False,
        chirality=Chirality.UNKNOWN,
        atoms=(1,),
    )
    monomer_b = Monomer(
        norine_name="Gly",
        residue="Gly",
        is_methylated=False,
        chirality=Chirality.UNKNOWN,
        atoms=(2,),
    )

    atom_info_dict = {
        1: Atom(name="C", hydrogens=0),
        2: Atom(name="N", hydrogens=0),
    }

    edge = AtomicEdge(1, 2)
    atomic_edges = {edge: AtomicBond(arity="1", bond_type="SINGLE")}

    mon_to_atom = {
        0: 1,
        1: 2,
    }

    # Since we use None for rules, it should use ALL_RULES.
    # No matches are expected because the dummy graphs won't match real rules,
    # but we verify that the function runs without throwing an exception.
    subtypes = classify_single_bond(
        source_monomer_idx=0,
        target_monomer_idx=1,
        monomer_a=monomer_a,
        monomer_b=monomer_b,
        mon_to_atom=mon_to_atom,
        atom_info_dict=atom_info_dict,
        atomic_edges=atomic_edges,
        rules=None,
    )
    assert isinstance(subtypes, list)
