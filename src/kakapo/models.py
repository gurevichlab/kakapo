"""Data models for kakapo.

Defines immutable data structures used throughout the application for type safety
and explicit contracts between modules.
"""

from dataclasses import dataclass, field, replace
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
        source_file_path: The path of the source MGF file from which this spectrum was loaded.
        peaks: List of detected peaks.
        precursor_mass: The m/z of the precursor ion (parent mass).
        charge: The charge of the precursor ion.
    """

    spectrum_id: str
    source_file_path: str
    peaks: frozenset[Peak]
    precursor_mass: float
    charge: int | None = None

    def __post_init__(self):
        if self.precursor_mass < 0:
            raise ValueError("Precursor mass must be non-negative")
        if not self.spectrum_id:
            raise ValueError("Spectrum ID cannot be empty")
        if len(self.peaks) == 0:
            raise ValueError("Spectrum must contain at least one peak")
        if not isinstance(self.source_file_path, str):
            raise TypeError("Source file path must be a string")
        if self.charge is not None and not isinstance(self.charge, int):
            raise TypeError("Charge must be an integer or None")

    def get_sorted_peaks(self) -> list[Peak]:
        """Return peaks sorted by mass."""
        return sorted(self.peaks, key=lambda p: p.mass)

    def get_neutral_mass(self) -> float:
        """Calculate the neutral precursor mass of the spectrum.

        Returns:
            The neutral precursor mass in Daltons.
        """
        from kakapo.constants import PROTON_MASS

        charge = self.charge if self.charge is not None else 1
        return self.precursor_mass * charge - PROTON_MASS * charge


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
        mask: Bitmask representing the monomers in this fragment.
        rules_shifts: Tuple of RuleShift instances applied to generate this ion.
        total_shift_mass: Sum of rules_shifts.shift.
    """

    fragment_id: str
    mask: int
    rules_shifts: tuple[RuleShift, ...] = ()
    total_shift_mass: float = field(init=False)

    def __post_init__(self):
        if not self.fragment_id:
            raise ValueError("Fragment ID cannot be empty")
        if not isinstance(self.mask, int):
            raise TypeError("Mask must be an integer")
        if self.mask <= 0:
            raise ValueError("Mask must be positive")
        object.__setattr__(self, "total_shift_mass", sum(item.shift for item in self.rules_shifts))


@dataclass(frozen=True)
class TheoreticalIonWithMass:
    """A theoretical ion with its mass calculated for a specific monomer mass vector.

    Attributes:
        fragment_id: Unique identifier for the fragment.
        mass: Calculated exact mass of this ion.
        mask: Bitmask representing the monomers in this fragment.
    """

    fragment_id: str
    mass: float
    mask: int

    def __post_init__(self):
        if not self.fragment_id:
            raise ValueError("Fragment ID cannot be empty")
        if self.mask <= 0:
            raise ValueError("Mask must be positive")


@dataclass(frozen=True)
class TheoreticalSpectrum:
    """A generated theoretical mass spectrum for a molecule.

    Attributes:
        structure_id: Unique identifier for the parent Molecule.
        ions: Frozenset of TheoreticalIon objects.
        structure_mass: Exact mass of the matched molecule.
        monomer_count: Number of monomers in the parent Molecule.
        monomer_mass_vector: Precomputed list of monomer masses.
    """

    structure_id: str
    ions: frozenset[TheoreticalIon]
    structure_mass: float
    monomer_count: int
    monomer_mass_vector: tuple[float, ...]

    def __post_init__(self):
        if not self.structure_id:
            raise ValueError("Structure ID cannot be empty")
        if len(self.ions) == 0:
            raise ValueError("Theoretical spectrum must contain at least one ion")
        if self.structure_mass < 0:
            raise ValueError("Structure mass must be non-negative")
        if self.monomer_count <= 0:
            raise ValueError("Monomer count must be positive")
        if not isinstance(self.monomer_mass_vector, tuple):
            raise TypeError("Monomer mass vector must be a tuple")
        if any(not isinstance(m, (int, float)) or m < 0 for m in self.monomer_mass_vector):
            raise ValueError("All monomer masses in vector must be non-negative numbers")
        if len(self.monomer_mass_vector) != self.monomer_count:
            raise ValueError("Monomer mass vector length must match monomer count")


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

        # Map current monomer keys to sequential 0-indexed keys
        sorted_keys = sorted(self.monomers.keys())
        if sorted_keys != list(range(len(self.monomers))):
            key_map = {old_k: i for i, old_k in enumerate(sorted_keys)}

            # Reindex monomers
            new_monomers = {key_map[k]: v for k, v in self.monomers.items()}
            object.__setattr__(self, "monomers", new_monomers)

            # Reindex bonds
            new_bonds = tuple(
                replace(
                    bond, source_idx=key_map[bond.source_idx], target_idx=key_map[bond.target_idx]
                )
                for bond in self.bonds
            )
            object.__setattr__(self, "bonds", new_bonds)

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
class MatchedPeak:
    """A theoretical peak matched to an experimental peak.

    Attributes:
        theoretical_mz: The m/z value of the theoretical peak.
        experimental_mz: The m/z value of the experimental peak.
        charge: The charge of the matched peak.
        theoretical_fragment_mask: Bitmask representing the monomers in this fragment.
    """

    theoretical_mz: float
    experimental_mz: float
    charge: int
    theoretical_fragment_mask: int

    def __post_init__(self):
        if self.theoretical_mz < 0:
            raise ValueError("Theoretical m/z must be non-negative")
        if self.experimental_mz < 0:
            raise ValueError("Experimental m/z must be non-negative")
        if self.charge <= 0:
            raise ValueError("Charge must be positive")
        if not isinstance(self.theoretical_fragment_mask, int):
            raise TypeError("Theoretical fragment mask must be an integer")
        if self.theoretical_fragment_mask <= 0:
            raise ValueError("Theoretical fragment mask must be positive")


@dataclass(frozen=True)
class MatchResult:
    """Result of matching a spectrum to a molecule.

    Attributes:
        structure_id: ID of the matched molecule.
        spectrum_id: ID of the matched spectrum.
        cleaned_spectrum: Cleaned experimental spectrum used for matching.
        structure_mass: Exact mass of the matched molecule.
        spectrum_mass: Precursor mass of the matched spectrum.
        mass_tolerance: Mass tolerance used for matching.
        matched_peaks_count: Number of experimental peaks matching theory.
        theoretical_peaks_count: Number of theoretical peaks generated.
        experimental_peaks_count: Number of experimental peaks in the spectrum.
        monomer_count: Number of monomers in the matched molecule.
        unmodified_monomer_mass_vector: Precalculated unmodified vector of monomer masses.
        matched_peaks: List of matched peaks.
        modified_monomer_idx: The 0-based index of the modified monomer in the best match.
        mass_mod: The calculated mass difference (precursor - structure) for the match.
        initial_modified_monomer_mass: The mass of the modified monomer before the mass
            modification was applied.
    """

    structure_id: str
    spectrum_id: str
    cleaned_spectrum: Spectrum
    structure_mass: float
    spectrum_mass: float
    mass_tolerance: float
    matched_peaks_count: int
    theoretical_peaks_count: int
    experimental_peaks_count: int
    monomer_count: int
    unmodified_monomer_mass_vector: tuple[float, ...]
    matched_peaks: list[MatchedPeak] = field(default_factory=list)
    modified_monomer_idx: int | None = None
    mass_mod: float | None = None
    initial_modified_monomer_mass: float | None = None

    def __post_init__(self):  # noqa: C901
        if self.structure_mass < 0:
            raise ValueError("Structure mass must be non-negative")
        if self.spectrum_mass < 0:
            raise ValueError("Spectrum mass must be non-negative")
        if self.mass_tolerance < 0:
            raise ValueError("Mass tolerance must be non-negative")
        if self.matched_peaks_count < 0:
            raise ValueError("Matched peaks count must be non-negative")
        if self.theoretical_peaks_count <= 0:
            raise ValueError("Theoretical peaks count must be positive")
        if self.experimental_peaks_count < 0:
            raise ValueError("Experimental peaks count must be non-negative")
        if self.monomer_count <= 0:
            raise ValueError("Monomer count must be positive")
        if not isinstance(self.unmodified_monomer_mass_vector, tuple):
            raise TypeError("Unmodified monomer mass vector must be a tuple")
        if any(not isinstance(m, (int, float)) or m < 0 for m in self.unmodified_monomer_mass_vector):
            raise ValueError("All monomer masses in vector must be non-negative numbers")
        if len(self.unmodified_monomer_mass_vector) != self.monomer_count:
            raise ValueError("Unmodified monomer mass vector length must match monomer count")
        if (
            self.initial_modified_monomer_mass is not None
            and self.initial_modified_monomer_mass < 0
        ):
            raise ValueError("Initial modified monomer mass must be non-negative")
        if not isinstance(self.cleaned_spectrum, Spectrum):
            raise TypeError("Cleaned spectrum must be a Spectrum instance")
