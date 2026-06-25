"""Bond subtype classification using RDKit C++ substructure matching.

Matches inter-monomer bonds against predefined SMARTS templates
to identify bond subtypes (e.g. amide_1) and their direction (reversed).
"""

from dataclasses import dataclass

from rdkit import Chem

from kakapo.models import Atom, AtomicBond, AtomicEdge, BondSubtype, Monomer
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)

# Predefined element-literal based SMARTS rules mapping to their corresponding subgraphs.
# Atoms with mapping numbers :1 and :2 represent the breakable bond connection.
SMARTS_RULES = {
    "amide_1": "[N:1](-[C:2]-O)-C",
    "amide_1_dnp": "[N:1](-[C:2]-O)-C",
    "amide_2": "N(-[C:1](-O)-[C:2])-C",
    "amide_3": "N(-[C:1](-S)-[C:2])-C",
    "amide_4": "[N:1](-[C:2]-O)(-C)-C",
    "amide_5": "[O:1](-[C:2]-O)-C",
    "amide_6": "N(-[C:1](-O)-[C:2])-C-C",
    "amide_7": "N(-[C:1](-O)-[C:2])-C-C",
    "amide_8": "N(-[C:1](-S)-[C:2])-C-C",
    "amide_9": "[N:1](-[C:2]-O)(-C)-N",
    "CC": "[C:1]-[C:2]",
    "NC": "[N:1]-[C:2]",
    "OC": "[O:1]-[C:2]",
}


@dataclass
class BondRule:
    """A chemical bond fragmentation rule represented as a SMARTS pattern.

    Attributes:
        name: The name of the rule/subtype (e.g., 'amide_1', 'NC').
        smarts: The SMARTS string representation.
        query_mol: The parsed RDKit query molecule.
        new_u: The RDKit atom index of mapped atom 1 (anchor u).
        new_v: The RDKit atom index of mapped atom 2 (anchor v).
    """

    name: str
    smarts: str
    query_mol: Chem.Mol
    new_u: int
    new_v: int


def make_rule(name: str, smarts: str) -> BondRule:
    """Compile a SMARTS pattern into a BondRule instance.

    Args:
        name: The name of the fragmentation rule.
        smarts: The SMARTS string.

    Returns:
        A fully constructed BondRule with cached query molecule and anchor indices.
    """
    query_mol = Chem.MolFromSmarts(smarts)
    if not query_mol:
        raise ValueError(f"Invalid SMARTS pattern: {smarts}")

    new_u = -1
    new_v = -1
    for atom in query_mol.GetAtoms():
        atom.SetNoImplicit(True)
        if atom.GetAtomMapNum() == 1:
            new_u = atom.GetIdx()
        elif atom.GetAtomMapNum() == 2:
            new_v = atom.GetIdx()

    if new_u == -1 or new_v == -1:
        raise ValueError(f"SMARTS pattern {name} does not contain map numbers 1 and 2")

    return BondRule(
        name=name,
        smarts=smarts,
        query_mol=query_mol,
        new_u=new_u,
        new_v=new_v,
    )


# Cached list of compiled rules
_RULES: list[BondRule] = []


def get_default_rules() -> list[BondRule]:
    """Retrieve the default list of fragmentation rules.

    Returns:
        A list of BondRule objects representing the default templates.
    """
    global _RULES
    if not _RULES:
        _RULES = [make_rule(name, smarts) for name, smarts in SMARTS_RULES.items()]
    return _RULES


def _build_local_mol(
    monomer_a: Monomer,
    monomer_b: Monomer,
    atom_info_dict: dict[int, Atom],
    atomic_edges: dict[AtomicEdge, AtomicBond],
) -> tuple[Chem.Mol, dict[int, int]]:
    """Helper method to construct a local RDKit Mol from the two monomers.

    Restricts the atoms and bonds to only those belonging to monomer_a or monomer_b,
    and returns the target molecule with cleared aromaticity and all single bonds.

    Args:
        monomer_a: The first Monomer instance.
        monomer_b: The second Monomer instance.
        atom_info_dict: Dictionary mapping atom IDs to Atom instances.
        atomic_edges: Dictionary mapping AtomicEdges to AtomicBonds.

    Returns:
        A tuple of (local_mol, id_to_rdkit_idx).
    """
    rw_mol = Chem.RWMol()
    id_to_rdkit_idx = {}
    relevant_atoms = set(monomer_a.atoms) | set(monomer_b.atoms)

    # Add relevant atom nodes in a sorted/reproducible order
    for atom_id in sorted(relevant_atoms):
        atom = atom_info_dict.get(atom_id)
        if atom is not None:
            rd_atom = Chem.Atom(atom.name)
            rd_atom.SetIsAromatic(False)
            rd_atom.SetFormalCharge(0)
            rd_atom.SetIsotope(0)
            rd_atom.SetNoImplicit(True)
            idx = rw_mol.AddAtom(rd_atom)
            id_to_rdkit_idx[atom_id] = idx

    # Add edges connecting these relevant atoms
    for edge in atomic_edges:
        if edge.atom1 in relevant_atoms and edge.atom2 in relevant_atoms:
            idx_u = id_to_rdkit_idx.get(edge.atom1)
            idx_v = id_to_rdkit_idx.get(edge.atom2)
            if idx_u is not None and idx_v is not None:
                rw_mol.AddBond(idx_u, idx_v, Chem.BondType.SINGLE)

    local_mol = rw_mol.GetMol()
    # Clear aromaticity and fix bond orders on final mol
    for bond in local_mol.GetBonds():
        bond.SetIsAromatic(False)
        bond.SetBondType(Chem.BondType.SINGLE)

    return local_mol, id_to_rdkit_idx


def _match_rule(
    local_mol: Chem.Mol,
    id_to_rdkit_idx: dict[int, int],
    rule: BondRule,
    edge: AtomicEdge,
) -> tuple[bool, bool]:
    """Helper method to execute RDKit substructure matching for a single rule.

    Determines if the rule's breakable edge maps to the connecting edge, and checks
    if the matched orientation is reversed compared to the monomer_a -> monomer_b direction.

    Args:
        local_mol: The local target Chem.Mol representing the monomer junction.
        id_to_rdkit_idx: Mapping from original atom ID to RDKit atom index.
        rule: The BondRule template to match.
        edge: The AtomicEdge connecting the two monomers.

    Returns:
        A tuple of (is_matched, is_reversed). If is_matched is False, is_reversed is False.
    """
    atom_a, atom_b = edge.atom1, edge.atom2
    if atom_a not in id_to_rdkit_idx or atom_b not in id_to_rdkit_idx:
        return False, False

    idx_a = id_to_rdkit_idx[atom_a]
    idx_b = id_to_rdkit_idx[atom_b]

    query_mol = rule.query_mol
    new_u = rule.new_u
    new_v = rule.new_v

    # 1. Quick element check (avoid running substructure match if elements do not match)
    rule_u_el = query_mol.GetAtomWithIdx(new_u).GetSymbol()
    rule_v_el = query_mol.GetAtomWithIdx(new_v).GetSymbol()

    atom_a_el = local_mol.GetAtomWithIdx(idx_a).GetSymbol()
    atom_b_el = local_mol.GetAtomWithIdx(idx_b).GetSymbol()

    forward_match = atom_a_el == rule_u_el and atom_b_el == rule_v_el
    reverse_match = atom_a_el == rule_v_el and atom_b_el == rule_u_el
    if not (forward_match or reverse_match):
        return False, False

    # 2. Substructure matching
    matches = local_mol.GetSubstructMatches(query_mol)
    for match in matches:
        mapped_u = match[new_u]
        mapped_v = match[new_v]

        # Check if the rule's breakable edge maps to our connecting atomic edge
        if mapped_u == idx_a and mapped_v == idx_b:
            return True, False
        elif mapped_u == idx_b and mapped_v == idx_a:
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
        rules: An optional list of loaded BondRules to match against. Defaults to get_default_rules().

    Returns:
        A list of BondSubtypes matched by the fragmentation rules.
    """
    if rules is None:
        rules = get_default_rules()

    local_mol, id_to_rdkit_idx = _build_local_mol(
        monomer_a, monomer_b, atom_info_dict, atomic_edges
    )

    edge = AtomicEdge(mon_to_atom[source_monomer_idx], mon_to_atom[target_monomer_idx])

    logger.debug(
        "Classifying single bond between monomers %d and %d using %d rules",
        source_monomer_idx,
        target_monomer_idx,
        len(rules),
    )

    subtypes = []
    for rule in rules:
        matched, is_reversed = _match_rule(local_mol, id_to_rdkit_idx, rule, edge)
        is_reversed = is_reversed ^ (
            mon_to_atom[source_monomer_idx] > mon_to_atom[target_monomer_idx]
        )
        if matched:
            logger.debug(
                "Matched subtype '%s' (reversed=%s) for monomer bond %d-%d",
                rule.name,
                is_reversed,
                source_monomer_idx,
                target_monomer_idx,
            )
            subtypes.append(BondSubtype(name=rule.name, reversed=is_reversed))

    return subtypes
