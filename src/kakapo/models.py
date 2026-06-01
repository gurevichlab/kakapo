"""Data models for kakapo.

Defines immutable data structures used throughout the application for type safety
and explicit contracts between modules.
"""

from dataclasses import dataclass
from enum import Enum, auto

import networkx as nx
from rban_parsing.rban_parser import Parsed_rBAN_Record


class Chirality(Enum):
    """Amino acid chirality."""

    L = auto()
    D = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class Peak:
    """A single mass spectrometry peak.

    Attributes:
        mass: The m/z value of the peak.
        intensity: The normalized intensity (0-1 or higher).
    """

    mass: float
    intensity: float

    def __post_init__(self):
        if self.mass < 0:
            raise ValueError("Peak mass must be non-negative")
        if self.intensity < 0:
            raise ValueError("Peak intensity must be non-negative")


@dataclass(frozen=True)
class Spectrum:
    """An experimental mass spectrometry spectrum.

    Attributes:
        spectrum_id: Unique identifier for this spectrum.
        peaks: List of detected peaks.
        precursor_mass: The m/z of the precursor ion (parent mass).
    """

    spectrum_id: str
    peaks: frozenset[Peak]
    precursor_mass: float

    def __post_init__(self):
        if self.precursor_mass < 0:
            raise ValueError("Precursor mass must be non-negative")
        if not self.spectrum_id:
            raise ValueError("Spectrum ID cannot be empty")
        if len(self.peaks) == 0:
            raise ValueError("Spectrum must contain at least one peak")

    def get_sorted_peaks(self) -> list[Peak]:
        """Return peaks sorted by mass."""
        return sorted(self.peaks, key=lambda p: p.mass)


@dataclass(frozen=True)
class TheoreticalIon:
    """A theoretical ion generated from a fragment.

    Attributes:
        fragment_id: Unique identifier for the source fragment (e.g., 'frag_0').
        mass: Calculated exact mass of the ion in Daltons.
        ion_type: The adduct or ion type (e.g., '[M+H]+').
    """

    fragment_id: str
    mass: float
    ion_type: str

    def __post_init__(self):
        if not self.fragment_id:
            raise ValueError("Fragment ID cannot be empty")
        if self.mass < 0:
            raise ValueError("Theoretical ion mass must be non-negative")
        if not self.ion_type:
            raise ValueError("Ion type cannot be empty")


@dataclass(frozen=True)
class TheoreticalSpectrum:
    """A generated theoretical mass spectrum for a molecule.

    Attributes:
        structure_id: Unique identifier for the parent Molecule.
        ions: Frozenset of TheoreticalIon objects.
    """

    structure_id: str
    ions: frozenset[TheoreticalIon]

    def __post_init__(self):
        if not self.structure_id:
            raise ValueError("Structure ID cannot be empty")
        if len(self.ions) == 0:
            raise ValueError("Theoretical spectrum must contain at least one ion")

    def get_sorted_ions(self) -> list[TheoreticalIon]:
        """Return ions sorted by mass."""
        return sorted(self.ions, key=lambda ion: ion.mass)


@dataclass(frozen=True)
class Monomer:
    """A single monomer (building block) in an rBAN structure.

    Attributes:
        norine_name: NORINE database name for the monomer.
        residue: The standard amino acid or core residue.
        is_methylated: Whether this monomer is methylated.
        chirality: L, D, or unknown stereochemistry.
    """

    norine_name: str
    residue: str
    is_methylated: bool
    chirality: Chirality

    def __post_init__(self):
        if not self.norine_name:
            raise ValueError("Monomer NORINE name cannot be empty")
        if not self.residue:
            raise ValueError("Monomer residue cannot be empty")


@dataclass(frozen=True)
class Bond:
    """A chemical bond between two monomers in an rBAN structure.

    Represents a linkage between adjacent monomers with explicit bond type
    (amino, ester, thioether, glycosidic, etc.) and information about whether
    it's part of a cyclic ring.

    Attributes:
        source_idx: Index of the source monomer in the parent Molecule's monomers tuple.
        target_idx: Index of the target monomer in the parent Molecule's monomers tuple.
        bond_type: String describing the bond type (e.g., 'AMINO', 'ESTER', 'THIOETHER').
        is_ring_bond: Whether this bond is part of a cyclic ring.
    """

    source_idx: int
    target_idx: int
    bond_type: str
    is_ring_bond: bool = False

    def __post_init__(self):
        if self.source_idx < 0 or self.target_idx < 0:
            raise ValueError("Bond indices must be non-negative")
        if self.source_idx == self.target_idx:
            raise ValueError("Bond cannot connect a monomer to itself")
        if not self.bond_type:
            raise ValueError("Bond type cannot be empty")


@dataclass(frozen=True)
class Cycle:
    """A cyclic peptide motif within an rBAN structure.

    Represents a ring of monomers connected in cyclic topology. Indices
    reference positions in the parent Molecule's monomers tuple, ordered
    as they appear in the cycle.

    Attributes:
        monomer_indices: Tuple of monomer indices forming this cycle.
    """

    monomer_indices: tuple[int, ...]

    def __post_init__(self):
        if len(self.monomer_indices) < 2:
            raise ValueError("A cycle must contain at least 2 monomers")
        if len(set(self.monomer_indices)) != len(self.monomer_indices):
            raise ValueError("Cycle monomer indices must be unique")


@dataclass(frozen=True)
class Molecule:
    """An rBAN molecular structure.

    Represents a complete natural product structure with monomers arranged
    in linear and/or cyclic topology. Cyclic regions are explicitly tracked
    to support fragmentation algorithms that split on bridges and 2-cuts.
    Bonds encode the chemical linkages between monomers.

    Attributes:
        structure_id: Unique identifier for this molecule.
        monomers: Ordered sequence of monomers in the structure.
        bonds: Chemical linkages between monomers (empty tuple for single monomers).
        cycles: Cyclic motifs within the structure (empty tuple for linear molecules).
        smiles: Optional SMILES string representation.
        source_parsed_record: Optional original Parsed_rBAN_Record payload for
            atom-level reconstruction (atoms, atomic bonds, monomer-to-atom mapping).
    """

    structure_id: str
    monomers: tuple[Monomer, ...]
    bonds: tuple[Bond, ...] = ()
    cycles: tuple[Cycle, ...] = ()
    smiles: str | None = None
    source_parsed_record: Parsed_rBAN_Record | None = None

    def __post_init__(self):
        if not self.structure_id:
            raise ValueError("Structure ID cannot be empty")
        if len(self.monomers) == 0:
            raise ValueError("Molecule must contain at least one monomer")

        # Validate bond indices
        monomer_count = len(self.monomers)
        for bond in self.bonds:
            if bond.source_idx < 0 or bond.source_idx >= monomer_count:
                raise ValueError(
                    f"Bond source index {bond.source_idx} out of range "
                    f"(molecule has {monomer_count} monomers)"
                )
            if bond.target_idx < 0 or bond.target_idx >= monomer_count:
                raise ValueError(
                    f"Bond target index {bond.target_idx} out of range "
                    f"(molecule has {monomer_count} monomers)"
                )

        # Validate cycle indices
        for cycle in self.cycles:
            for idx in cycle.monomer_indices:
                if idx < 0 or idx >= monomer_count:
                    raise ValueError(
                        f"Cycle contains invalid monomer index {idx} "
                        f"(molecule has {monomer_count} monomers)"
                    )

    @property
    def monomer_count(self) -> int:
        """Return the number of monomers."""
        return len(self.monomers)

    @property
    def is_cyclic(self) -> bool:
        """Return True if the molecule contains any cyclic regions."""
        return len(self.cycles) > 0

    def get_monomer_cycles(self, monomer_index: int) -> list[Cycle]:
        """Get all cycles containing a specific monomer.

        Args:
            monomer_index: Index of the monomer in the monomers tuple.

        Returns:
            List of Cycle objects containing this monomer (typically 0 or 1).

        Raises:
            IndexError: If monomer_index is out of bounds.
        """
        if monomer_index < 0 or monomer_index >= len(self.monomers):
            raise IndexError(f"Monomer index {monomer_index} out of bounds")

        return [cycle for cycle in self.cycles if monomer_index in cycle.monomer_indices]

    def is_monomer_cyclic(self, monomer_index: int) -> bool:
        """Check if a monomer participates in any cycle.

        Args:
            monomer_index: Index of the monomer in the monomers tuple.

        Returns:
            True if the monomer is in a cycle, False otherwise.

        Raises:
            IndexError: If monomer_index is out of bounds.
        """
        return bool(self.get_monomer_cycles(monomer_index))

    def get_linear_segments(self) -> list[tuple[int, int]]:
        """Get ranges of consecutive linear (non-cyclic) monomers.

        Returns:
            List of (start_idx, end_idx) tuples for linear regions (inclusive).
            Useful for the fragmenter to identify segments that can be
            cleanly fragmented by bridge cuts.
        """
        cyclic_indices = set()
        for cycle in self.cycles:
            cyclic_indices.update(cycle.monomer_indices)

        segments = []
        start = None

        for i in range(len(self.monomers)):
            if i not in cyclic_indices:
                if start is None:
                    start = i
            else:
                if start is not None:
                    segments.append((start, i - 1))
                    start = None

        if start is not None:
            segments.append((start, len(self.monomers) - 1))

        return segments

    def build_networkx_graph(self) -> nx.DiGraph:
        """Build a NetworkX directed graph representation of the molecular structure.

        Nodes represent monomers (labeled by their index), and edges represent bonds.
        Edge attributes include 'bond_type' and 'is_ring_bond'.

        Returns:
            A directed graph with monomer indices as nodes and bonds as edges.
            Each edge (u, v) has attributes: bond_type (str) and is_ring_bond (bool).

        Example:
            >>> mol = Molecule(...)
            >>> G = mol.build_networkx_graph()
            >>> G.number_of_nodes()  # Number of monomers
            >>> for u, v, data in G.edges(data=True):
            ...     print(f"Bond {u}->{v}: {data['bond_type']}")
        """
        G = nx.DiGraph()

        # Add nodes for each monomer
        for i in range(len(self.monomers)):
            G.add_node(i)

        # Add edges for each bond
        for bond in self.bonds:
            G.add_edge(
                bond.source_idx,
                bond.target_idx,
                bond_type=bond.bond_type,
                is_ring_bond=bond.is_ring_bond,
            )

        return G


@dataclass(frozen=True)
class MatchResult:
    """Result of matching a spectrum to a molecule.

    Attributes:
        molecule_id: ID of the matched molecule.
        spectrum_id: ID of the matched spectrum.
        score: Numeric score (typically 0-1 or higher).
        matched_peaks_count: Number of experimental peaks matching theory.
        theoretical_peaks_count: Number of theoretical peaks generated.
        monomer_count: Number of monomers in the matched molecule.
    """

    molecule_id: str
    spectrum_id: str
    score: float
    matched_peaks_count: int
    theoretical_peaks_count: int
    monomer_count: int

    def __post_init__(self):
        if self.score < 0:
            raise ValueError("Score must be non-negative")
        if self.matched_peaks_count < 0:
            raise ValueError("Matched peaks count must be non-negative")
        if self.theoretical_peaks_count <= 0:
            raise ValueError("Theoretical peaks count must be positive")
        if self.monomer_count <= 0:
            raise ValueError("Monomer count must be positive")
