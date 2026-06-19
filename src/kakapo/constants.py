"""Static scientific constants for kakapo.

This module stores immutable chemistry and mass-spec constants that are loaded
and parsed on import from external configuration files.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

# Locate constants files relative to this file
CONSTANTS_DIR = Path(__file__).parent / "constants"
ELEMENT_LIST_PATH = CONSTANTS_DIR / "element_list.txt"
SIGN_STANDARD_PATH = CONSTANTS_DIR / "sign_standard.txt"


def calculate_residue_mass_from_smiles(
    smiles: str,
    atomic_masses: dict[str, float],
    water_mass: float,
) -> float:
    """Calculate the exact residue mass from a SMILES string.

    The SMILES represents a free amino acid. Its residue mass is calculated as
    the monoisotopic mass of the SMILES molecule minus the exact mass of water (H2O).

    Args:
        smiles: The SMILES string.
        atomic_masses: Dictionary mapping element symbols to monoisotopic masses.
        water_mass: Exact mass of water (H2O) in Daltons.

    Returns:
        The exact residue mass in Daltons.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: '{smiles}'")
    # Add implicit/explicit hydrogens to ensure correct H count
    mol = Chem.AddHs(mol)

    total_mass = 0.0
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atomic_masses:
            total_mass += atomic_masses[symbol]
        else:
            # Fallback to standard atomic weight if symbol not in element_list
            total_mass += atom.GetMass()

    return total_mass - water_mass


def load_atomic_masses(path: Path) -> dict[str, float]:
    """Parse exact atomic masses from element_list.txt."""
    masses = {}
    if not path.exists():
        raise FileNotFoundError(f"Atomic masses configuration file not found at {path}")

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                symbol = parts[0]
                try:
                    mass = float(parts[1])
                    masses[symbol] = mass
                except ValueError:
                    continue
    return masses


def load_amino_acid_masses(
    path: Path,
    atomic_masses: dict[str, float],
    water_mass: float,
) -> dict[str, float]:
    """Parse and calculate residue masses from sign_standard.txt."""
    aa_masses = {}
    if not path.exists():
        raise FileNotFoundError(f"Amino acid standards file not found at {path}")

    with open(path, encoding="utf-8") as f:
        # Skip header line
        f.readline()
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                # Format: amino-acid PubChem-ID Norine-name Formula Smiles
                # E.g. "ala 5950 Ala C3H7NO2 A A 0 CC(N)C(O)=O"
                # Use uppercase abbreviation as key to match python constants style
                aa_name = parts[0].upper()
                smiles = parts[-1]
                try:
                    aa_masses[aa_name] = calculate_residue_mass_from_smiles(
                        smiles,
                        atomic_masses,
                        water_mass,
                    )
                except Exception as e:
                    # Provide useful warning and raise error if critical
                    raise ValueError(
                        f"Failed to calculate residue mass for {aa_name} "
                        f"from SMILES '{smiles}': {e}"
                    ) from e
    return aa_masses


# Load configurations on import
try:
    ATOMIC_MASSES = load_atomic_masses(ELEMENT_LIST_PATH)
except Exception as e:
    print(f"Error loading atomic masses: {e}", file=sys.stderr)
    ATOMIC_MASSES = {}

# Use parsed exact hydrogen and oxygen masses to calculate water mass and proton mass
H_MASS = ATOMIC_MASSES.get("H", 1.007825)
PROTON_MASS = 1.007276
WATER_EXACT_MASS = 2 * ATOMIC_MASSES.get("H", 1.007825) + ATOMIC_MASSES.get("O", 15.994915)
MAX_CHARGE = 2

try:
    AMINO_ACID_MASSES = load_amino_acid_masses(
        SIGN_STANDARD_PATH,
        ATOMIC_MASSES,
        WATER_EXACT_MASS,
    )
except Exception as e:
    print(f"Error loading amino acid residue masses: {e}", file=sys.stderr)
    AMINO_ACID_MASSES = {}


@dataclass(frozen=True)
class FragmentationRule:
    """A chemical fragmentation rule for molecular graph cleavage.

    Defines the eligible bond type and its corresponding left/right mass shifts.

    Attributes:
        bond_type: The string representing the eligible bond type (e.g. 'PEPTIDE').
        left_shift: Mass shift applied to the target monomer of the cleaved bond (N-side).
        right_shift: Mass shift applied to the source monomer of the cleaved bond (C-side).
    """

    bond_type: str
    left_shift: float
    right_shift: float


# All predefined rules from npdtools config files
DEFAULT_RULES = (
    FragmentationRule("amide_1", H_MASS, -H_MASS),
    FragmentationRule("amide_2", H_MASS, -H_MASS),
    FragmentationRule("amide_3", H_MASS, -H_MASS),
    FragmentationRule("amide_4", H_MASS, -H_MASS),
    FragmentationRule("amide_5", H_MASS, -H_MASS),
    FragmentationRule("amide_6", H_MASS, -H_MASS),
    FragmentationRule("amide_7", H_MASS, -H_MASS),
    FragmentationRule("amide_8", H_MASS, -H_MASS),
    FragmentationRule("amide_9", H_MASS, -H_MASS),
    FragmentationRule("NC", H_MASS, -H_MASS),
    FragmentationRule("OC", H_MASS, -H_MASS),
    FragmentationRule("CC", 0, 0),
    # FragmentationRule("PEPTIDE", H_MASS, -H_MASS),
    # FragmentationRule("AMINO", H_MASS, -H_MASS),
)
