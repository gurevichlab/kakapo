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

from dataclasses import dataclass

import networkx as nx
from rban_parsing.rban_parser import Parsed_rBAN_Record

from kakapo.constants import AMINO_ACID_MASSES, ATOMIC_MASSES, PROTON_MASS
from kakapo.models import Bond, Molecule, Monomer

# Type aliases for clarity
Edge = tuple[int, int]  # Undirected edge between two monomers (canonicalized)
EdgePair = tuple[Edge, Edge]  # Pair of edges (for 2-cuts in cycles)


@dataclass(frozen=True)
class Fragment:
    """A single molecular fragment resulting from bond cleavage.

    Represents a connected subset of monomers extracted from the parent molecule
    by removing specific bonds (e.g., bridges or 2-cut edges).

    Attributes:
        monomer_indices: Ordered tuple of monomer indices in this fragment.
        removed_bonds: Bonds that were cleaved to generate this fragment.
        is_from_cycle: Whether this fragment originated from a cyclic region.
    """

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

    def __init__(self, molecule: Molecule):
        """Initialize fragmentation graph from a molecule.

        Args:
            molecule: A kakapo Molecule instance.
        """
        self.molecule = molecule
        # Keep the original directed graph for metadata compatibility, but use an
        # undirected view for connectivity-based fragmentation operations
        self.graph = molecule.build_networkx_graph()
        self.undirected_graph = self.graph.to_undirected()

    def find_all_bridges(self) -> list[Edge]:
        """Find all bridges (cut edges) in the molecular graph.

        A bridge is an edge whose removal increases the number of connected
        components. In molecular terms, cutting a bridge severs the structure.

        Returns:
            List of (vertex, vertex) edge tuples representing undirected bridges.
        """
        bridges = []
        for u, v in nx.bridges(self.undirected_graph):
            bridges.append(_canonical_edge((u, v)))
        return bridges

    def find_all_cycles(self) -> list[list[int]]:
        """Find all simple cycles in the molecular graph.

        Returns:
            List of node sequences representing distinct cycles.
        """
        return list(nx.cycle_basis(self.undirected_graph))

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
            return []  # Need at least 3 nodes to have distinct 2-cuts

        pairs = []
        n = len(cycle_nodes)

        # Generate all pairs of edge indices (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Edge from node i to next node in cycle
                edge1 = _canonical_edge((cycle_nodes[i], cycle_nodes[(i + 1) % n]))
                # Edge from node j to next node in cycle
                edge2 = _canonical_edge((cycle_nodes[j], cycle_nodes[(j + 1) % n]))
                # Avoid adding duplicate pairs (same edges in different order)
                if edge1 != edge2:
                    pairs.append((edge1, edge2))

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
    """Return orientation-independent edge representation."""
    u, v = edge
    return (u, v) if u <= v else (v, u)


class MonomerMassCalculator:
    """Lazy-loading calculator for monomer exact masses.

    Uses atomic composition from Parsed_rBAN_Record when available for exact
    mass calculation. Falls back to RDKit SMILES or amino acid lookup table.
    Results are cached to avoid repeated calculations.

    Attributes:
        monomers: List of Monomer instances indexed by their position.
        parsed_record: Optional Parsed_rBAN_Record for atomic-level mass calculation.
        _mass_cache: Internal cache mapping monomer index -> mass (Daltons).
    """

    def __init__(
        self,
        monomers: list[Monomer],
        parsed_record: Parsed_rBAN_Record | None = None,
    ):
        """Initialize mass calculator for a set of monomers.

        Args:
            monomers: List of Monomer instances from a Molecule.
            parsed_record: Optional Parsed_rBAN_Record for atomic composition access.
        """
        self.monomers = monomers
        self.parsed_record = parsed_record
        self._mass_cache: dict[int, float] = {}

    def get_mass(self, monomer_idx: int) -> float:
        """Get exact molecular mass of a monomer.

        Tries mass calculation in order:
        1. Atomic composition from Parsed_rBAN_Record (if available)
        2. RDKit calculation from SMILES (if available)
        3. Amino acid residue lookup table (fallback)

        Args:
            monomer_idx: Index of the monomer in the monomers list.

        Returns:
            Exact mass in Daltons.

        Raises:
            IndexError: If monomer_idx is out of bounds.
            ValueError: If mass cannot be determined.
        """
        if monomer_idx < 0 or monomer_idx >= len(self.monomers):
            raise IndexError(f"Monomer index {monomer_idx} out of bounds")

        if monomer_idx in self._mass_cache:
            return self._mass_cache[monomer_idx]

        # Try atomic composition calculation (most accurate)
        mass = self._calculate_mass_from_atoms(monomer_idx)

        # Try RDKit calculation from SMILES (if available in future)
        if mass is None:
            mass = self._calculate_mass_rdk(monomer_idx)

        # Fallback: use amino acid lookup table
        if mass is None:
            mass = self._estimate_mass_from_residue(self.monomers[monomer_idx].residue)

        if mass is None:
            raise ValueError(
                f"Cannot determine mass for monomer {self.monomers[monomer_idx].norine_name}"
            )

        self._mass_cache[monomer_idx] = mass
        return mass

    def _calculate_mass_from_atoms(self, monomer_idx: int) -> float | None:
        """Calculate exact mass from atomic composition.

        Uses the atomic graph stored in Parsed_rBAN_Record to compute the sum
        of atomic masses for all atoms belonging to this monomer, including
        implicit hydrogens.

        Args:
            monomer_idx: Index of the monomer in the monomers list.

        Returns:
            Exact mass in Daltons, or None if source record unavailable.

        Raises:
            ValueError: If atom type not found in ATOMIC_MASSES.
        """
        if self.parsed_record is None:
            return None

        # Get monomer info from parsed record (mapping 0-indexed position to sorted original keys)
        ordered_keys = sorted(self.parsed_record.monomers.keys())
        if 0 <= monomer_idx < len(ordered_keys):
            parsed_key = ordered_keys[monomer_idx]
            monomer_info = self.parsed_record.monomers.get(parsed_key)
        else:
            monomer_info = None

        if monomer_info is None:
            return None

        total_mass = 0.0

        # Sum masses of all atoms in this monomer
        for atom_id in monomer_info.atoms:
            atom_info = self.parsed_record.atoms.get(atom_id)
            if atom_info is None:
                return None

            # Get atomic mass by element symbol
            element_symbol = atom_info.name  # e.g., "C", "N", "O", "S"
            if element_symbol not in ATOMIC_MASSES:
                raise ValueError(
                    f"Unknown element '{element_symbol}' for atom {atom_id} "
                    f"in monomer {monomer_idx}"
                )

            # Add mass of the atom
            total_mass += ATOMIC_MASSES[element_symbol]

            # Add mass of implicit hydrogens
            if atom_info.hydrogens > 0:
                total_mass += atom_info.hydrogens * ATOMIC_MASSES["H"]

        return total_mass if total_mass > 0 else None

    def _calculate_mass_rdk(self, monomer_idx: int) -> float | None:
        """Calculate mass using RDKit (placeholder for future enhancement).

        Args:
            monomer_idx: Index of the monomer.

        Returns:
            Exact mass in Daltons, or None if SMILES unavailable.
        """
        # TODO: Extract SMILES from monomer/variant metadata and use RDKit
        # For now, return None to trigger fallback
        return None

    def _estimate_mass_from_residue(self, residue: str) -> float | None:
        """Look up approximate mass from standard residue name.

        Args:
            residue: Standard residue name (e.g., 'ALA', 'GLY').

        Returns:
            Approximate mass in Daltons, or None if not found.
        """
        # Try uppercase
        if residue.upper() in AMINO_ACID_MASSES:
            return AMINO_ACID_MASSES[residue.upper()]

        # Try first 3 chars (common abbreviation)
        abbrev = residue[:3].upper()
        if abbrev in AMINO_ACID_MASSES:
            return AMINO_ACID_MASSES[abbrev]

        return None


def generate_fragments(molecule: Molecule) -> list[Fragment]:
    """Generate all possible fragments from a molecule via bridge and 2-cut cleavage.

    Algorithm:
    1. Build a graph from the molecule's bonds
    2. For each bridge: remove it to create linear fragments
    3. For each cycle: enumerate all 2-cut combinations to create cyclic fragments
    4. Return unique fragments

    Args:
        molecule: A kakapo Molecule instance.

    Returns:
        List of Fragment objects representing all distinct molecular fragments.

    Example:
        >>> mol = Molecule(structure_id="test", monomers=(m1, m2, m3), bonds=...)
        >>> fragments = generate_fragments(mol)
        >>> print(f"Generated {len(fragments)} fragments")
    """
    if len(molecule.monomers) == 1:
        # Single monomer: no cleavage possible
        return [Fragment(monomer_indices=(0,), removed_bonds=())]

    graph = FragmentationGraph(molecule)
    fragments = []
    seen_fragments = set()

    # Phase 1: Bridge cuts (linear fragmentation)
    bridges = graph.find_all_bridges()

    if bridges:
        for bridge in bridges:
            components = graph.get_connected_components_after_removal([bridge])
            for comp in components:
                comp_tuple = tuple(sorted(comp))
                if comp_tuple not in seen_fragments:
                    # Find bonds that were removed to create this component
                    removed_bonds = _find_bonds_for_component(molecule, comp)
                    fragment = Fragment(
                        monomer_indices=comp_tuple,
                        removed_bonds=tuple(removed_bonds),
                        is_from_cycle=False,
                    )
                    fragments.append(fragment)
                    seen_fragments.add(comp_tuple)

    # Phase 2: 2-cut enumeration (cyclic fragmentation)
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
                        monomer_indices=comp_tuple,
                        removed_bonds=tuple(removed_bonds),
                        is_from_cycle=True,
                    )
                    fragments.append(fragment)
                    seen_fragments.add(comp_tuple)

    return fragments


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

        if src_in != tgt_in:  # Bond crosses the boundary
            removed_bonds.append(bond)
        elif _canonical_edge((bond.source_idx, bond.target_idx)) in removed_edge_set:
            removed_bonds.append(bond)

    return removed_bonds


def calculate_theoretical_masses(
    fragment: Fragment,
    calculator: MonomerMassCalculator,
) -> list[tuple[float, str]]:
    """Calculate theoretical ion masses from a fragment.

    Generates masses for three ion types:
    1. [M+H]+: protonated molecular ion
    2. [M-H]-: deprotonated molecular ion
    3. Neutral fragment (for MS/MS fragments)

    Mass calculations:
    - Sum monomer exact masses
    - Add/subtract proton mass (1.007825 Da) for charged ions

    Args:
        fragment: Fragment instance to calculate masses for.
        calculator: MonomerMassCalculator for monomer mass lookups.

    Returns:
        List of (mass, ion_type) tuples. Masses are floats in Daltons.
        Ion types are: '[M+H]+', '[M-H]-', 'fragment'.

    Raises:
        ValueError: If mass cannot be calculated for any monomer.
    """
    # Sum monomer masses
    monomer_mass_sum = sum(calculator.get_mass(idx) for idx in fragment.monomer_indices)

    # Neutral fragment mass (Dereplicator-style: no per-cleavage water subtraction)
    neutral_mass = monomer_mass_sum

    # Ion masses
    masses = [
        (neutral_mass + PROTON_MASS, "[M+H]+"),
        (neutral_mass - PROTON_MASS, "[M-H]-"),
        (neutral_mass, "fragment"),
    ]

    return masses


def generate_all_theoretical_masses(
    molecule: Molecule,
    calculator: MonomerMassCalculator | None = None,
) -> list[tuple[str, float, str]]:
    """Generate all theoretical masses from a molecule.

    Convenience function that:
    1. Generates all fragments from the molecule
    2. Calculates masses for each fragment using atomic composition when available
    3. Returns flattened list of (fragment_id, mass, ion_type) tuples

    Args:
        molecule: The Molecule to fragment and analyze.
        calculator: Optional MonomerMassCalculator. If None, creates one with
            atomic mass calculation enabled (if source_parsed_record available).

    Returns:
        List of (fragment_id, mass, ion_type) tuples sorted by mass.
        Fragment IDs are like 'frag_0', 'frag_1', etc.

    Example:
        >>> mol = Molecule(...)
        >>> masses = generate_all_theoretical_masses(mol)
        >>> for frag_id, mass, ion_type in masses[:10]:
        ...     print(f"{frag_id}: {mass:.4f} ({ion_type})")
    """
    if calculator is None:
        calculator = MonomerMassCalculator(
            list(molecule.monomers),
            parsed_record=molecule.source_parsed_record,
        )

    fragments = generate_fragments(molecule)
    all_masses = []

    for frag_idx, fragment in enumerate(fragments):
        frag_id = f"frag_{frag_idx}"
        masses = calculate_theoretical_masses(fragment, calculator)
        for mass, ion_type in masses:
            all_masses.append((frag_id, mass, ion_type))

    # Sort by mass
    all_masses.sort(key=lambda x: x[1])

    return all_masses
