"""Theoretical fragment generation from rBAN structures.

This module implements the core fragmentation logic for generating theoretical
mass spectra from rBAN molecules. It handles:
- Breaking molecular graphs at bridge bonds (single cuts)
- Enumerating 2-cut combinations in cyclic regions
- Calculating fragment masses from monomer totals
- Generating theoretical ions ([M+H]+, [M-H]-, neutral fragments)

Key algorithm: Graph-theoretic approach using bridge detection (for linear parts)
and 2-cut enumeration (for cyclic parts).
"""

from dataclasses import dataclass, replace

import networkx as nx

from kakapo.constants import (
    DEFAULT_RULES,
    FragmentationRule,
)
from kakapo.models import Bond, Molecule, Monomer, RuleShift, TheoreticalIon, TheoreticalSpectrum
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)

# Type aliases for clarity
Edge = tuple[int, int]  # Undirected edge between two monomers (canonicalized)
EdgePair = tuple[Edge, Edge]  # Pair of edges (for 2-cuts in cycles)


@dataclass(frozen=True)
class Fragment:
    """A single molecular fragment resulting from bond cleavage.

    Represents a connected subset of monomers extracted from the parent molecule
    by removing specific bonds (e.g., bridges or 2-cut edges).

    Attributes:
        fragment_id: Unique identifier for the generated fragment.
        monomer_indices: Ordered tuple of monomer indices in this fragment.
        removed_bonds: Bonds that were cleaved to generate this fragment.
        is_from_cycle: Whether this fragment originated from a cyclic region.
    """

    fragment_id: str
    monomer_indices: tuple[int, ...]
    removed_bonds: tuple[Bond, ...]
    is_from_cycle: bool = False

    def __post_init__(self):
        if len(self.monomer_indices) == 0:
            raise ValueError("Fragment must contain at least one monomer")
        if len(set(self.monomer_indices)) != len(self.monomer_indices):
            raise ValueError("Fragment monomer indices must be unique")

    @property
    def monomer_count(self) -> int:
        """Return the number of monomers in this fragment."""
        return len(self.monomer_indices)


class FragmentationGraph:
    """Helper class for graph-based molecular fragmentation.

    Wraps a Molecule's bond structure as a NetworkX directed graph and provides
    methods to identify bridges and cycles for systematic fragmentation.

    Attributes:
        molecule: The parent Molecule being fragmented.
        graph: NetworkX DiGraph representation (nodes = monomer indices).
    """

    def __init__(self, molecule: Molecule, rules: tuple[FragmentationRule, ...] = DEFAULT_RULES):
        """Initialize fragmentation graph from a molecule.

        Args:
            molecule: A kakapo Molecule instance.
            rules: Tuple of FragmentationRules allowed for cleavage.
                Defaults to DEFAULT_RULES.
        """
        self.molecule = molecule
        self.rules = rules
        self.graph = molecule.build_networkx_graph()
        self.undirected_graph = self.graph.to_undirected()

    def _is_bond_breakable(self, edge: Edge) -> bool:
        """Check if an undirected edge corresponds to a breakable bond.

        Args:
            edge: The canonical Edge representing a bond.

        Returns:
            True if the bond has an active fragmentation rule, False otherwise.
        """
        for bond in self.molecule.bonds:
            if _canonical_edge((bond.source_idx, bond.target_idx)) == _canonical_edge(edge):
                # A bond is breakable if its bond_type matches a rule
                # OR if one of its subtypes matches a rule!
                if any(rule.bond_type == bond.bond_type for rule in self.rules):
                    return True
                if any(any(rule.bond_type == s.name for s in bond.subtypes) for rule in self.rules):
                    return True
        return False

    def find_all_bridges(self) -> list[Edge]:
        """Find all bridges (cut edges) in the molecular graph.

        A bridge is an edge whose removal increases the number of connected
        components. In molecular terms, cutting a bridge severs the structure.

        Returns:
            List of (vertex, vertex) edge tuples representing undirected bridges.
        """
        bridges = []
        for u, v in nx.bridges(self.undirected_graph):
            edge = _canonical_edge((u, v))
            if self._is_bond_breakable(edge):
                bridges.append(edge)
        logger.debug("Found %d breakable bridge(s) in molecular graph", len(bridges))
        return bridges

    def find_all_cycles(self) -> list[list[int]]:
        """Find all simple cycles in the molecular graph.

        Returns:
            List of node sequences representing distinct cycles.
        """
        cycles = list(nx.cycle_basis(self.undirected_graph))
        logger.debug("Found %d cycle(s) in molecular graph", len(cycles))
        return cycles

    def extract_2cut_pairs(self, cycle_nodes: list[int]) -> list[EdgePair]:
        """Extract all 2-cut edge pairs from a cycle.

        A 2-cut is a pair of edges whose removal disconnects a cycle into
        linear fragments. For a cycle of length n, there are C(n,2) = n(n-1)/2
        possible 2-cut combinations.

        Args:
            cycle_nodes: List of node indices forming a cycle.

        Returns:
            List of ((src1, tgt1), (src2, tgt2)) edge pair tuples.
        """
        if len(cycle_nodes) < 3:
            logger.debug(
                "Skipping 2-cut extraction for cycle with < 3 nodes: %s",
                cycle_nodes,
            )
            return []  # Need at least 3 nodes to have distinct 2-cuts

        pairs = []
        n = len(cycle_nodes)

        # Generate all pairs of edge indices (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                edge1 = _canonical_edge((cycle_nodes[i], cycle_nodes[(i + 1) % n]))
                edge2 = _canonical_edge((cycle_nodes[j], cycle_nodes[(j + 1) % n]))
                if edge1 != edge2:
                    if self._is_bond_breakable(edge1) and self._is_bond_breakable(edge2):
                        pairs.append((edge1, edge2))

        logger.debug(
            "Extracted %d breakable 2-cut edge pair(s) from cycle %s",
            len(pairs),
            cycle_nodes,
        )
        return pairs

    def get_connected_components_after_removal(
        self,
        removed_edges: list[Edge],
    ) -> list[list[int]]:
        """Extract connected components after removing specified edges.

        Args:
            removed_edges: List of (source, target) edges to remove.

        Returns:
            List of connected components (each as a list of node indices).
        """
        G_copy = self.undirected_graph.copy()

        for edge in removed_edges:
            src, tgt = _canonical_edge(edge)
            if G_copy.has_edge(src, tgt):
                G_copy.remove_edge(src, tgt)

        components = list(nx.connected_components(G_copy))
        return [sorted(comp) for comp in components]


def _canonical_edge(edge: Edge) -> Edge:
    """Return orientation-independent edge representation.

    Args:
        edge: The input edge tuple.

    Returns:
        The edge tuple sorted by monomer index.
    """
    u, v = edge
    return (u, v) if u <= v else (v, u)


def _process_bridge_cuts(
    molecule: Molecule,
    graph: FragmentationGraph,
    seen_fragments: set[tuple[int, ...]],
) -> list[Fragment]:
    """Cleave bridges in the molecular graph to generate linear fragments.

    Args:
        molecule: The parent Molecule instance.
        graph: The FragmentationGraph helper instance.
        seen_fragments: Set of already processed fragment index tuples to avoid duplicates.

    Returns:
        List of generated Fragment objects from bridge cuts.
    """
    fragments = []
    bridges = graph.find_all_bridges()
    for bridge in bridges:
        components = graph.get_connected_components_after_removal([bridge])
        for comp in components:
            comp_tuple = tuple(sorted(comp))
            if comp_tuple not in seen_fragments:
                removed_bonds = _find_bonds_for_component(molecule, comp)
                fragment = Fragment(
                    fragment_id="",
                    monomer_indices=comp_tuple,
                    removed_bonds=tuple(removed_bonds),
                    is_from_cycle=False,
                )
                fragments.append(fragment)
                seen_fragments.add(comp_tuple)
    return fragments


def _process_cycle_cuts(
    molecule: Molecule,
    graph: FragmentationGraph,
    seen_fragments: set[tuple[int, ...]],
) -> list[Fragment]:
    """Enumerate 2-cuts in cycles to generate cyclic fragments.

    Args:
        molecule: The parent Molecule instance.
        graph: The FragmentationGraph helper instance.
        seen_fragments: Set of already processed fragment index tuples to avoid duplicates.

    Returns:
        List of generated Fragment objects from cycle 2-cuts.
    """
    fragments = []
    cycles = graph.find_all_cycles()
    for cycle in cycles:
        if len(cycle) < 3:
            continue  # Skip 2-node cycles (no valid 2-cut)

        pairs = graph.extract_2cut_pairs(cycle)
        for edge_pair in pairs:
            components = graph.get_connected_components_after_removal(list(edge_pair))
            for comp in components:
                comp_tuple = tuple(sorted(comp))
                if comp_tuple not in seen_fragments:
                    removed_bonds = _find_bonds_for_component(molecule, comp, list(edge_pair))
                    fragment = Fragment(
                        fragment_id="",
                        monomer_indices=comp_tuple,
                        removed_bonds=tuple(removed_bonds),
                        is_from_cycle=True,
                    )
                    fragments.append(fragment)
                    seen_fragments.add(comp_tuple)
    return fragments


def generate_fragments(
    molecule: Molecule,
    rules: tuple[FragmentationRule, ...] = DEFAULT_RULES,
) -> list[Fragment]:
    """Generate all possible fragments from a molecule via bridge and 2-cut cleavage.

    Algorithm:
    1. Build a graph from the molecule's bonds
    2. For each bridge: remove it to create linear fragments
    3. For each cycle: enumerate all 2-cut combinations to create cyclic fragments
    4. Return unique fragments

    Args:
        molecule: A kakapo Molecule instance.
        rules: Tuple of FragmentationRules allowed for cleavage.
            Defaults to DEFAULT_RULES.

    Returns:
        List of Fragment objects representing all distinct molecular fragments.

    Example:
        >>> mol = Molecule(structure_id="test", monomers=(m1, m2, m3), bonds=...)
        >>> fragments = generate_fragments(mol)
        >>> print(f"Generated {len(fragments)} fragments")
    """
    logger.info("Generating fragments for molecule '%s'", molecule.structure_id)
    if len(molecule.monomers) == 1:
        # Single monomer: no cleavage possible
        single_idx = next(iter(molecule.monomers.keys()))
        logger.debug(
            "Molecule '%s' has 1 monomer. No cuts possible.", molecule.structure_id
        )
        return [Fragment(fragment_id="frag_0", monomer_indices=(single_idx,), removed_bonds=())]

    graph = FragmentationGraph(molecule, rules=rules)
    seen_fragments = set()

    # Phase 1: Bridge cuts
    fragments = _process_bridge_cuts(molecule, graph, seen_fragments)
    logger.debug(
        "Molecule '%s': generated %d fragments from bridge cuts",
        molecule.structure_id,
        len(fragments),
    )

    # Phase 2: 2-cut enumeration
    cycle_fragments = _process_cycle_cuts(molecule, graph, seen_fragments)
    logger.debug(
        "Molecule '%s': generated %d fragments from cycle cuts",
        molecule.structure_id,
        len(cycle_fragments),
    )
    fragments.extend(cycle_fragments)

    # Reconstruct fragments to assign unique IDs in order
    final_fragments = []
    for idx, fragment in enumerate(fragments):
        final_fragments.append(replace(fragment, fragment_id=f"frag_{idx}"))

    logger.info(
        "Generated %d theoretical fragments for molecule '%s'",
        len(final_fragments),
        molecule.structure_id,
    )
    return final_fragments


def _find_bonds_for_component(
    molecule: Molecule,
    component: list[int],
    removed_edges: list[Edge] | None = None,
) -> list[Bond]:
    """Find bonds that were cleaved to generate a specific component.

    Helper function to identify which bonds were removed from the parent
    to create a disconnected component.

    Args:
        molecule: The parent Molecule.
        component: List of monomer indices in the resulting component.
        removed_edges: Specific edges that were removed (optional).

    Returns:
        List of Bond objects that were cleaved.
    """
    if removed_edges is None:
        removed_edges = []

    removed_bonds = []
    component_set = set(component)
    removed_edge_set = {_canonical_edge(edge) for edge in removed_edges}

    for bond in molecule.bonds:
        # Check if this bond connects component to outside or is explicitly removed
        src_in = bond.source_idx in component_set
        tgt_in = bond.target_idx in component_set

        if src_in != tgt_in:
            removed_bonds.append(bond)
        elif _canonical_edge((bond.source_idx, bond.target_idx)) in removed_edge_set:
            removed_bonds.append(bond)

    return removed_bonds


def _calculate_bond_cleavage_shift(
    bond: Bond,
    rules: tuple[FragmentationRule, ...],
    fragment_monomers: set[int],
) -> RuleShift | None:
    """Calculate the rule name and mass shift for a single cleaved bond relative to a fragment.

    Args:
        bond: The cleaved Bond.
        rules: Tuple of FragmentationRules.
        fragment_monomers: Set of monomer indices in the fragment.

    Returns:
        A RuleShift object in Daltons, or None if no rule matches.
    """
    for rule in rules:
        matching_subtype = next((s for s in bond.subtypes if s.name == rule.bond_type), None)
        if matching_subtype is not None or bond.bond_type == rule.bond_type:
            is_reversed = matching_subtype.reversed if matching_subtype is not None else False
            left_shift = rule.left_shift
            right_shift = rule.right_shift

            shift = 0.0
            target_in = bond.target_idx in fragment_monomers
            source_in = bond.source_idx in fragment_monomers

            if is_reversed:
                if target_in:
                    shift += left_shift
                if source_in:
                    shift += right_shift
            else:
                if target_in:
                    shift += right_shift
                if source_in:
                    shift += left_shift
            return RuleShift(rule_name=rule.bond_type, shift=shift)
    return None


def calculate_theoretical_ion(
    fragment: Fragment,
    monomers: dict[int, Monomer],
    rules: tuple[FragmentationRule, ...] = DEFAULT_RULES,
    fragment_id: str | None = None,
) -> TheoreticalIon:
    """Calculate the theoretical ion for a fragment, including bond cleavage shifts.

    Args:
        fragment: Fragment instance to calculate the ion for.
        monomers: dict[int, Monomer] for monomer mass lookups.
        rules: Tuple of FragmentationRules allowed for cleavage.
            Defaults to DEFAULT_RULES.
        fragment_id: Unique identifier for the generated ion. If None, uses fragment.fragment_id.

    Returns:
        A TheoreticalIon object.

    Raises:
        IndexError: If monomer index in the fragment is out of bounds.
        ValueError: If mass cannot be calculated for any monomer.
    """
    if fragment_id is None:
        fragment_id = fragment.fragment_id
    logger.debug("Calculating theoretical mass for fragment '%s'", fragment_id)
    # Sum monomer exact masses
    monomer_mass_sum = 0.0
    for idx in fragment.monomer_indices:
        if idx not in monomers:
            logger.error("Monomer index %d out of bounds for fragment %s", idx, fragment_id)
            raise IndexError(f"Monomer index {idx} out of bounds")
        monomer = monomers[idx]
        if monomer.mass is None:
            logger.error("Cannot determine mass for monomer %s", monomer.norine_name)
            raise ValueError(f"Cannot determine mass for monomer {monomer.norine_name}")
        monomer_mass_sum += monomer.mass

    # Apply bond-specific shifts from the removed boundary bonds
    fragment_monomers = set(fragment.monomer_indices)
    rules_shifts = []
    for bond in fragment.removed_bonds:
        rule_shift = _calculate_bond_cleavage_shift(bond, rules, fragment_monomers)
        if rule_shift is not None:
            rules_shifts.append(rule_shift)

    total_shift = sum(item.shift for item in rules_shifts)
    ion_mass = monomer_mass_sum + total_shift

    logger.debug(
        "Fragment '%s': monomer mass sum = %f, cleavage shifts = %f, ion mass = %f",
        fragment_id,
        monomer_mass_sum,
        total_shift,
        ion_mass,
    )

    return TheoreticalIon(
        fragment_id=fragment_id,
        mass=ion_mass,
        rules_shifts=tuple(rules_shifts),
    )


def generate_all_theoretical_masses(
    molecule: Molecule,
    rules: tuple[FragmentationRule, ...] = DEFAULT_RULES,
) -> list[TheoreticalIon]:
    """Generate all theoretical masses from a molecule.

    Convenience function that:
    1. Generates all fragments from the molecule
    2. Calculates masses for each fragment using atomic composition when available
    3. Returns flattened list of TheoreticalIon objects

    Args:
        molecule: The Molecule to fragment and analyze.
        rules: Tuple of FragmentationRules allowed for cleavage.
            Defaults to DEFAULT_RULES.

    Returns:
        List of TheoreticalIon objects sorted by mass.
    """
    fragments = generate_fragments(molecule, rules=rules)
    all_masses = []

    for fragment in fragments:
        ion = calculate_theoretical_ion(
            fragment, molecule.monomers, rules=rules, fragment_id=fragment.fragment_id
        )
        all_masses.append(ion)

    # Sort by mass
    all_masses.sort(key=lambda x: x.mass)

    return all_masses


def generate_theoretical_spectrum(
    molecule: Molecule,
    rules: tuple[FragmentationRule, ...] = DEFAULT_RULES,
) -> TheoreticalSpectrum:
    """Generate the theoretical spectrum for a molecule.

    Args:
        molecule: The Molecule to fragment and analyze.
        rules: Tuple of FragmentationRules allowed for cleavage.
            Defaults to DEFAULT_RULES.

    Returns:
        TheoreticalSpectrum object representing the predicted ions.
    """
    all_masses = generate_all_theoretical_masses(
        molecule,
        rules=rules,
    )

    structure_mass = 0.0
    for monomer in molecule.monomers.values():
        if monomer.mass is None:
            raise ValueError(
                f"Cannot calculate structure mass: monomer '{monomer.norine_name}' has no mass"
            )
        structure_mass += monomer.mass

    return TheoreticalSpectrum(
        structure_id=molecule.structure_id,
        ions=frozenset(all_masses),
        structure_mass=structure_mass,
        monomer_count=molecule.monomer_count,
    )
