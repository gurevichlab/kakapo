"""Bond subtype classification using subgraph isomorphism.

Matches inter-monomer bonds against NPDtools fragmentation rule MOL files
to identify bond subtypes (e.g. amide_1) and their direction (reversed).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from networkx.algorithms import isomorphism
from rdkit import Chem

from kakapo.models import Atom, AtomicBond, AtomicEdge, BondSubtype, Monomer

logger = logging.getLogger(__name__)

# Static mapping of rule names to their breakable bond indices
BREAKABLE_BOND_INDICES = {
    "amide_1": 0,
    "amide_1_dnp": 0,
    "amide_2": 2,
    "amide_3": 2,
    "amide_4": 0,
    "amide_5": 0,
    "amide_6": 2,
    "amide_7": 2,
    "amide_8": 2,
    "amide_9": 0,
    "CC": 0,
    "NC": 0,
    "OC": 0,
}


@dataclass
class BondRule:
    """A chemical bond fragmentation rule parsed from configuration.

    This rule matches local chemical subgraphs to detect specific structural linkages
    (such as amide or general carbon/heteroatom bonds) and define their fragmentation characteristics.

    Attributes:
        name: The name of the rule/subtype (e.g., 'amide_1', 'NC').
        mol_path: The file path to the V3000 `.mol` representation of the rule.
        mol_graph: NetworkX representation of the rule's chemical graph (excluding hydrogen).
        mol_breakable_edge: The tuple of atom indices in `mol_graph` defining the breakable bond.
    """

    name: str
    mol_path: str
    mol_graph: nx.Graph
    mol_breakable_edge: tuple[int, int]


def load_rules(rule_dir_path: str | Path) -> list[BondRule]:
    """Parse all rules directly from the specified directory.

    Reads each V3000 `.mol` file from the directory, filters out hydrogen atoms (which are
    explicit in the .mol rule files but implicit in standard parsed records), maps their
    designated breakable bond indices, and constructs directed matching templates.

    Args:
        rule_dir_path: The directory path to load rules from.

    Returns:
        A list of BondRule objects representing the parsed and constructed rule templates.
    """
    rule_dir = Path(rule_dir_path)
    rules = []
    if not rule_dir.exists():
        logger.warning(f"Rules directory {rule_dir} does not exist.")
        return rules

    for mol_path in rule_dir.glob("*.mol"):
        name = mol_path.stem
        # Get breakable bond index keyed by rule name (default 0)
        bond_idx = BREAKABLE_BOND_INDICES.get(name, 0)

        mol = Chem.MolFromMolFile(str(mol_path), removeHs=False)
        if not mol:
            logger.debug(f"Failed to parse MOL file {mol_path}")
            continue

        # Get breakable edge connecting atom_u and atom_v
        try:
            breakable_bond = mol.GetBondWithIdx(bond_idx)
            mol_breakable_edge = (
                breakable_bond.GetBeginAtomIdx(),
                breakable_bond.GetEndAtomIdx(),
            )
        except (IndexError, RuntimeError):
            logger.warning(f"Bond index {bond_idx} out of range in MOL {mol_path.name}")
            continue

        # Build NetworkX representation of rule MOL (without H atoms)
        mol_graph = nx.Graph()
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "H":
                mol_graph.add_node(atom.GetIdx(), element=atom.GetSymbol())

        # Add edges between non-H atoms
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            if (
                mol.GetAtomWithIdx(u).GetSymbol() != "H"
                and mol.GetAtomWithIdx(v).GetSymbol() != "H"
            ):
                mol_graph.add_edge(u, v)

        rules.append(
            BondRule(
                name=name,
                mol_path=str(mol_path),
                mol_graph=mol_graph,
                mol_breakable_edge=mol_breakable_edge,
            )
        )
    return rules


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
AMIDE_RULES_DIR = PACKAGE_ROOT.joinpath("constants", "Fragmentation_rule", "Amide_rule")
GENERAL_RULES_DIR = PACKAGE_ROOT.joinpath("constants", "Fragmentation_rule", "General_rule")

# Dynamic rule loading with caching to prevent import-time I/O
ALL_RULES: list[BondRule] = []
_RULES_LOADED = False


def get_default_rules() -> list[BondRule]:
    """Retrieve the default list of fragmentation rules, loading them if not already cached.

    Returns:
        A list of BondRule objects representing the default templates.
    """
    global ALL_RULES, _RULES_LOADED
    # Respect any external mocking/patching of ALL_RULES in tests
    if not _RULES_LOADED and not ALL_RULES:
        default_rules = load_rules(AMIDE_RULES_DIR) + load_rules(GENERAL_RULES_DIR)
        ALL_RULES.clear()
        ALL_RULES.extend(default_rules)
        _RULES_LOADED = True
    return ALL_RULES


def _build_local_graph(
    monomer_a: Monomer,
    monomer_b: Monomer,
    atom_info_dict: dict[int, Atom],
    atomic_edges: dict[AtomicEdge, AtomicBond],
) -> nx.Graph:
    """Helper method to construct a local NetworkX Graph from the two monomers.

    Restricts the nodes to only those atoms belonging to monomer_a or monomer_b,
    and includes all atomic bonds connecting these relevant atoms.

    Args:
        monomer_a: The first Monomer instance.
        monomer_b: The second Monomer instance.
        atom_info_dict: Dictionary mapping atom IDs to Atom instances.
        atomic_edges: Dictionary mapping AtomicEdges to AtomicBonds.

    Returns:
        A NetworkX Graph representing the local atomic structure.
    """
    local_graph = nx.Graph()
    relevant_atoms = set(monomer_a.atoms) | set(monomer_b.atoms)

    # Add relevant atom nodes
    for atom_id in relevant_atoms:
        atom = atom_info_dict.get(atom_id)
        if atom is not None:
            local_graph.add_node(atom_id, element=atom.name)

    # Add edges connecting these relevant atoms
    for edge in atomic_edges:
        if edge.atom1 in relevant_atoms and edge.atom2 in relevant_atoms:
            local_graph.add_edge(edge.atom1, edge.atom2)

    return local_graph


def _match_rule(
    local_graph: nx.Graph,
    rule: BondRule,
    edge: AtomicEdge,
) -> tuple[bool, bool]:
    """Helper method to execute GraphMatcher for a single rule on the local graph.

    Determines if the rule's breakable edge maps to the connecting edge, and checks
    if the matched orientation is reversed compared to the monomer_a -> monomer_b direction.

    Args:
        local_graph: The local atomic graph of the two monomers.
        rule: The BondRule template to match.
        edge: The AtomicEdge connecting the two monomers.

    Returns:
        A tuple of (is_matched, is_reversed). If is_matched is False, is_reversed is False.
    """

    def node_match(n1, n2):
        return n1.get("element") == n2.get("element")

    atom_a, atom_b = edge.atom1, edge.atom2
    GM = isomorphism.GraphMatcher(local_graph, rule.mol_graph, node_match=node_match)

    for sub in GM.subgraph_isomorphisms_iter():
        inv_sub = {v: k for k, v in sub.items()}
        u, v = rule.mol_breakable_edge

        mapped_u = inv_sub.get(u)
        mapped_v = inv_sub.get(v)

        # Check if the rule's breakable edge maps to our connecting atomic edge
        if mapped_u == atom_a and mapped_v == atom_b:
            # Matches the directed source -> target orientation
            return True, False
        elif mapped_u == atom_b and mapped_v == atom_a:
            # Matches in reverse
            return True, True

    return False, False


def classify_single_bond(
    source_monomer_idx: int,
    target_monomer_idx: int,
    monomer_a: Monomer,
    monomer_b: Monomer,
    mon_to_atom: dict[int, int],
    atom_info_dict: dict[int, Atom],
    atomic_edges: dict[AtomicEdge, AtomicBond],
    rules: list[BondRule] | None = None,
) -> list[BondSubtype]:
    """Classifies a single inter-monomer bond by matching its local subgraph against predefined
    rules.

    Restricts matching to the atoms and bonds within the two monomers connected by the edge.

    Args:
        source_monomer_idx: The index of the source monomer.
        target_monomer_idx: The index of the target monomer.
        monomer_a: The source Monomer instance of the bond.
        monomer_b: The target Monomer instance of the bond.
        mon_to_atom: Mapping from monomer ID to the atom ID connecting to the other monomer.
        atom_info_dict: Dictionary mapping atom IDs to Atom instances.
        atomic_edges: Dictionary mapping AtomicEdge instances to AtomicBond instances.
        rules: An optional list of loaded BondRules to match against. Defaults to ALL_RULES.

    Returns:
        A list of BondSubtypes matched by the fragmentation rules.
    """
    if rules is None:
        rules = get_default_rules()

    local_graph = _build_local_graph(monomer_a, monomer_b, atom_info_dict, atomic_edges)

    edge = AtomicEdge(mon_to_atom[source_monomer_idx], mon_to_atom[target_monomer_idx])

    subtypes = []
    for rule in rules:
        matched, is_reversed = _match_rule(local_graph, rule, edge)
        is_reversed = is_reversed ^ (
            mon_to_atom[source_monomer_idx] > mon_to_atom[target_monomer_idx]
        )
        if matched:
            subtypes.append(BondSubtype(name=rule.name, reversed=is_reversed))

    return subtypes
