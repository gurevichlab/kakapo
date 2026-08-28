"""Theoretical fragment generation from rBAN structures.

This module implements the core fragmentation logic for generating theoretical
ions from molecules. It handles:
- Building fragmentation graphs and identifying breakable bonds based on rules.
- Cleaving bridge bonds (1-cuts) in the molecular graph.
- Cleaving pairs of bonds (2-cuts) in cyclic regions of the molecular graph.
- Calculating rule-based mass shifts from bond cleavages for each fragment.
- Creating theoretical ions (TheoreticalIon) with monomer masks and mass shifts.
- Constructing theoretical spectra (TheoreticalSpectrum) containing the generated ions.

Note:
    Specific charge states and adducts (e.g., [M+H]+) are not generated directly in this
    module; instead, this module produces base theoretical ions (TheoreticalIon) with
    cleavage mass shifts. Adduct/charge state calculation is handled during spectrum
    construction and indexing.

Key algorithm: Graph-theoretic approach using bridge detection (for linear parts)
and 2-cut enumeration (for cyclic parts) based on optimal XOR-hashing.
"""

from dataclasses import dataclass, replace

from kakapo.constants import (
    DEFAULT_RULES,
    FragmentationRule,
)
from kakapo.core.algorithms import BridgeCut, TwoCut, find_bridges_and_2cuts
from kakapo.models import Bond, Molecule, RuleShift, TheoreticalIon, TheoreticalSpectrum
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
        mask: Bitmask representing the monomers in this fragment.
        removed_bonds: Bonds that were cleaved to generate this fragment.
        is_from_cycle: Whether this fragment originated from a cyclic region.
    """

    fragment_id: str
    mask: int
    removed_bonds: tuple[Bond, ...]
    is_from_cycle: bool = False

    def __post_init__(self):
        if self.mask == 0:
            raise ValueError("Fragment must contain at least one monomer")

    @property
    def monomer_count(self) -> int:
        """Return the number of monomers in this fragment."""
        return self.mask.bit_count()


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
        self.cut_result = find_bridges_and_2cuts(self.undirected_graph)

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

    def find_all_bridges(self) -> list[BridgeCut]:
        """Find all bridges (cut edges) in the molecular graph.

        A bridge is an edge whose removal increases the number of connected
        components. In molecular terms, cutting a bridge severs the structure.

        Returns:
            List of BridgeCut objects representing breakable bridges.
        """
        bridges = []
        for bridge in self.cut_result.bridges:
            if self._is_bond_breakable(bridge.edge):
                bridges.append(bridge)
        logger.debug("Found %d breakable bridge(s) in molecular graph", len(bridges))
        return bridges

    def get_breakable_2cuts(self) -> list[TwoCut]:
        """Filter the precomputed 2-cuts to keep only those where both edges are breakable.

        Returns:
            List of TwoCut objects.
        """
        breakable_2cuts = []
        for two_cut in self.cut_result.two_cuts:
            edge1, edge2 = two_cut.edge_pair
            if self._is_bond_breakable(edge1) and self._is_bond_breakable(edge2):
                breakable_2cuts.append(two_cut)
        logger.debug(
            "Filtered %d breakable 2-cut edge pair(s) from %d raw 2-cuts",
            len(breakable_2cuts),
            len(self.cut_result.two_cuts),
        )
        return breakable_2cuts


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
) -> list[Fragment]:
    """Cleave bridges in the molecular graph to generate linear fragments.

    Args:
        molecule: The parent Molecule instance.
        graph: The FragmentationGraph helper instance.

    Returns:
        List of generated Fragment objects from bridge cuts.
    """
    fragments = []
    bridges = graph.find_all_bridges()
    for bridge in bridges:
        removed_bonds = tuple(_find_bonds_for_component(molecule, [bridge.edge]))
        for mask in (bridge.mask1, bridge.mask2):
            fragment = Fragment(
                fragment_id="",
                mask=mask,
                removed_bonds=removed_bonds,
                is_from_cycle=False,
            )
            fragments.append(fragment)
    return fragments


def _process_cycle_cuts(
    molecule: Molecule,
    graph: FragmentationGraph,
) -> list[Fragment]:
    """Enumerate 2-cuts in cycles to generate cyclic fragments.

    Args:
        molecule: The parent Molecule instance.
        graph: The FragmentationGraph helper instance.

    Returns:
        List of generated Fragment objects from cycle 2-cuts.
    """
    fragments = []
    breakable_2cuts = graph.get_breakable_2cuts()
    for two_cut in breakable_2cuts:
        removed_bonds = tuple(_find_bonds_for_component(molecule, list(two_cut.edge_pair)))
        for mask in (two_cut.mask1, two_cut.mask2):
            fragment = Fragment(
                fragment_id="",
                mask=mask,
                removed_bonds=removed_bonds,
                is_from_cycle=True,
            )
            fragments.append(fragment)
    return fragments


def generate_fragments(
    molecule: Molecule,
    rules: tuple[FragmentationRule, ...] = DEFAULT_RULES,
) -> list[Fragment]:
    """Generate all possible fragments from a molecule via bridge and 2-cut cleavage.

    Algorithm:
    1. Build a graph from the molecule's bonds.
    2. Run the optimal XOR-hashing algorithm to identify cuts.
    3. For each breakable bridge: yield fragments directly from mask partitions.
    4. For each breakable 2-cut: yield fragments directly from mask partitions.
    5. Return unique fragments.

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
    logger.debug("Generating fragments for molecule '%s'", molecule.structure_id)
    graph = FragmentationGraph(molecule, rules=rules)

    if len(molecule.monomers) == 1:
        # Single monomer: no cleavage possible
        single_idx = next(iter(molecule.monomers.keys()))
        logger.debug("Molecule '%s' has 1 monomer. No cuts possible.", molecule.structure_id)
        return [Fragment(fragment_id="frag_0", mask=(1 << single_idx), removed_bonds=())]

    # Phase 1: Bridge cuts
    fragments = _process_bridge_cuts(molecule, graph)
    logger.debug(
        "Molecule '%s': generated %d fragments from bridge cuts",
        molecule.structure_id,
        len(fragments),
    )

    # Phase 2: 2-cut enumeration
    cycle_fragments = _process_cycle_cuts(molecule, graph)
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

    logger.debug(
        "Generated %d theoretical fragments for molecule '%s'",
        len(final_fragments),
        molecule.structure_id,
    )
    return final_fragments


def _find_bonds_for_component(
    molecule: Molecule,
    removed_edges: list[Edge],
) -> list[Bond]:
    """Find bonds that correspond to the removed edges.

    Args:
        molecule: The parent Molecule.
        removed_edges: The edges that were removed to create the cut.

    Returns:
        List of Bond objects that correspond to the removed edges.
    """
    removed_set = {_canonical_edge(edge) for edge in removed_edges}
    return [
        bond
        for bond in molecule.bonds
        if _canonical_edge((bond.source_idx, bond.target_idx)) in removed_set
    ]


def _calculate_bond_cleavage_shift(
    bond: Bond,
    rules: tuple[FragmentationRule, ...],
    mask: int,
) -> RuleShift | None:
    """Calculate the rule name and mass shift for a single cleaved bond relative to a fragment.

    Args:
        bond: The cleaved Bond.
        rules: Tuple of FragmentationRules.
        mask: Bitmask representing the component monomers.

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
            target_in = bool((mask >> bond.target_idx) & 1)
            source_in = bool((mask >> bond.source_idx) & 1)

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
    masses_vector: list[float],
    rules: tuple[FragmentationRule, ...] = DEFAULT_RULES,
    fragment_id: str | None = None,
) -> TheoreticalIon:
    """Calculate the theoretical ion for a fragment, including bond cleavage shifts.

    Args:
        fragment: Fragment instance to calculate the ion for.
        masses_vector: Precomputed list of monomer masses aligned with bits.
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

    total_valid_mask = (1 << len(masses_vector)) - 1
    if (fragment.mask & ~total_valid_mask) != 0:
        raise IndexError("Monomer index out of bounds")

    monomer_mass_sum = 0.0
    mask = fragment.mask
    idx = 0
    while mask > 0:
        if mask & 1:
            monomer_mass_sum += masses_vector[idx]
        mask >>= 1
        idx += 1

    # Apply bond-specific shifts from the removed boundary bonds
    rules_shifts = []
    for bond in fragment.removed_bonds:
        rule_shift = _calculate_bond_cleavage_shift(bond, rules, fragment.mask)
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
        mask=fragment.mask,
        rules_shifts=tuple(rules_shifts),
    )


def generate_all_theoretical_masses(
    molecule: Molecule,
    rules: tuple[FragmentationRule, ...] = DEFAULT_RULES,
    masses_vector: list[float] | None = None,
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
        masses_vector: Optional precomputed list of monomer masses.

    Returns:
        List of TheoreticalIon objects.
    """
    if masses_vector is None:
        # Build masses_vector directly aligned with 0-indexed monomers
        masses_vector = [0.0] * len(molecule.monomers)
        for node, monomer in molecule.monomers.items():
            if monomer.mass is None:
                raise ValueError(f"Cannot determine mass for monomer {monomer.norine_name}")
            masses_vector[node] = monomer.mass

    fragments = generate_fragments(molecule, rules=rules)
    all_masses = []

    for fragment in fragments:
        ion = calculate_theoretical_ion(
            fragment,
            masses_vector=masses_vector,
            rules=rules,
            fragment_id=fragment.fragment_id,
        )
        all_masses.append(ion)

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
    masses_vector = [0.0] * len(molecule.monomers)
    for node, monomer in molecule.monomers.items():
        if monomer.mass is None:
            raise ValueError(
                f"Cannot calculate structure mass: monomer '{monomer.norine_name}' has no mass"
            )
        masses_vector[node] = monomer.mass

    all_masses = generate_all_theoretical_masses(
        molecule,
        rules=rules,
        masses_vector=masses_vector,
    )

    structure_mass = sum(masses_vector)

    return TheoreticalSpectrum(
        structure_id=molecule.structure_id,
        ions=frozenset(all_masses),
        structure_mass=structure_mass,
        monomer_count=molecule.monomer_count,
        monomer_mass_vector=tuple(masses_vector),
        monomers=molecule.monomers,
    )
