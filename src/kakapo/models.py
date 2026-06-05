"""Data models for kakapo.

Defines immutable data structures used throughout the application for type safety
and explicit contracts between modules.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import networkx as nx
from rban_parsing.rban_parser import NRP_metadata, Parsed_rBAN_Record


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
class RuleShift:
    """A mass shift applied by a specific fragmentation rule.

    Attributes:
        rule_name: The name of the fragmentation rule (bond type).
        shift: The mass shift value in Daltons.
    """

    rule_name: str
    shift: float


@dataclass(frozen=True)
class TheoreticalIon:
    """A theoretical ion generated from a fragment.

    Attributes:
        fragment_id: Unique identifier for the source fragment (e.g., 'frag_0').
        mass: Calculated exact mass of the ion in Daltons.
        rules_shifts: Tuple of RuleShift instances applied to generate this ion.
    """

    fragment_id: str
    mass: float
    rules_shifts: tuple[RuleShift, ...] = ()

    def __post_init__(self):
        if not self.fragment_id:
            raise ValueError("Fragment ID cannot be empty")
        if self.mass < 0:
            raise ValueError("Theoretical ion mass must be non-negative")


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
class Atom:
    """A chemical atom represented at the atomic detail level.

    Attributes:
        name: The chemical element symbol (e.g., 'C', 'N', 'O').
        hydrogens: The number of implicit hydrogen atoms attached to this atom.
    """

    name: str
    hydrogens: int


@dataclass(frozen=True)
class AtomicEdge:
    """An undirected bond edge between two atoms.

    Automatically canonicalizes indices upon instantiation such that atom1 <= atom2.
    This enables direction-agnostic comparison and hashing, allowing it to be used
    as a key in dictionaries or sets representing atomic bond networks.

    Attributes:
        atom1: The smaller atom ID in the bond.
        atom2: The larger atom ID in the bond.
    """

    atom1: int
    atom2: int

    def __post_init__(self):
        """Canonicalize the atom IDs to ensure atom1 <= atom2."""
        if self.atom1 > self.atom2:
            a1 = self.atom2
            a2 = self.atom1
            object.__setattr__(self, "atom1", a1)
            object.__setattr__(self, "atom2", a2)


@dataclass(frozen=True)
class AtomicBond:
    """An atomic bond containing chemical details of a connection.

    Attributes:
        arity: The bond order or arity (e.g., '1' for single, '2' for double, 'ar' for aromatic).
        bond_type: The qualitative chemical bond type or classification.
    """
    arity: str
    bond_type: str


@dataclass(frozen=True)
class BondSubtype:
    """A classified chemical subtype assigned to an inter-monomer bond.

    This encodes specific chemical connectivity patterns (e.g., 'amide_1' through 'amide_9',
    'NC', 'OC') matching NPDtools fragmentation rules. It includes orientation details
    relative to the configuration's reference subgraph.

    Attributes:
        name: The name of the matched fragmentation subtype (e.g., 'amide_1').
        reversed: True if the bond's source-to-target direction in the molecule
            (monomer1 -> monomer2) is opposite to the matched rule's directed breakable bond.
            False otherwise.
    """

    name: str
    reversed: bool


@dataclass(frozen=True)
class Monomer:
    """A single monomer (building block) in an rBAN structure.

    Attributes:
        norine_name: NORINE database name for the monomer.
        residue: The standard amino acid or core residue.
        is_methylated: Whether this monomer is methylated.
        chirality: L, D, or unknown stereochemistry.
        atoms: Tuple of atom IDs belonging to this monomer.
        mass: Precalculated monoisotopic mass of the monomer.
    """

    norine_name: str
    residue: str
    is_methylated: bool
    chirality: Chirality
    atoms: tuple[int, ...] = ()
    mass: float | None = None

    def __post_init__(self):
        if not self.norine_name:
            raise ValueError("Monomer NORINE name cannot be empty")
        if not self.residue:
            raise ValueError("Monomer residue cannot be empty")


@dataclass(frozen=True)
class Bond:
    """A chemical bond between two monomers in an rBAN structure.

    Represents a linkage between adjacent monomers with explicit bond type
    (amino, ester, thioether, glycosidic, etc.).

    Attributes:
        source_idx: Index of the source monomer in the parent Molecule's monomers dict.
        target_idx: Index of the target monomer in the parent Molecule's monomers dict.
        bond_type: String describing the bond type (e.g., 'AMINO', 'ESTER', 'THIOETHER').
        subtypes: Tuple of classified bond subtypes (e.g., 'amide_1').
    """

    source_idx: int
    target_idx: int
    bond_type: str
    subtypes: tuple[BondSubtype, ...] = ()

    def __post_init__(self):
        if self.source_idx < 0 or self.target_idx < 0:
            raise ValueError("Bond indices must be non-negative")
        if self.source_idx == self.target_idx:
            raise ValueError("Bond cannot connect a monomer to itself")
        if not self.bond_type:
            raise ValueError("Bond type cannot be empty")


@dataclass(frozen=True)
class Molecule:
    """An rBAN molecular structure.

    Represents a complete natural product structure with monomers arranged
    in linear and/or cyclic topology. Bonds encode the chemical linkages between monomers.

    Attributes:
        structure_id: Unique identifier for this molecule.
        monomers: Mapping of monomer IDs to Monomer instances.
        bonds: Chemical linkages between monomers (empty tuple for single monomers).
        smiles: Optional SMILES string representation.
        atoms: Mapping of atom IDs to Atom instances.
        atomic_bonds: Mapping of atomic edges to AtomicBond instances.
        metadata: Optional metadata.
        source_parsed_record: Optional original Parsed_rBAN_Record payload.
    """

    structure_id: str
    monomers: dict[int, Monomer]
    bonds: tuple[Bond, ...] = ()
    smiles: str | None = None
    atoms: dict[int, Atom] = field(default_factory=dict)
    atomic_bonds: dict[AtomicEdge, AtomicBond] = field(default_factory=dict)
    metadata: NRP_metadata | None = None
    source_parsed_record: Parsed_rBAN_Record | None = None

    def __post_init__(self):
        if not self.structure_id:
            raise ValueError("Structure ID cannot be empty")
        if len(self.monomers) == 0:
            raise ValueError("Molecule must contain at least one monomer")

        # Validate bond indices
        for bond in self.bonds:
            if bond.source_idx not in self.monomers:
                raise ValueError(f"Bond source index {bond.source_idx} not in monomers")
            if bond.target_idx not in self.monomers:
                raise ValueError(f"Bond target index {bond.target_idx} not in monomers")

    @property
    def monomer_count(self) -> int:
        """Return the number of monomers."""
        return len(self.monomers)

    def build_networkx_graph(self) -> nx.DiGraph:
        """Build a NetworkX directed graph representation of the molecular structure.

        Nodes represent monomers, and edges represent bonds.

        Returns:
            A directed graph.
        """
        G = nx.DiGraph()

        # Add nodes for each monomer
        for key in self.monomers.keys():
            G.add_node(key)

        # Add edges for each bond
        for bond in self.bonds:
            G.add_edge(
                bond.source_idx,
                bond.target_idx,
                bond_type=bond.bond_type,
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
