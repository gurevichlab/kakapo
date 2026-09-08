"""Static scientific constants for kakapo.

This module stores immutable chemistry and mass-spec constants that are loaded
and parsed on import from external configuration files.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Locate constants files relative to this file
CONSTANTS_DIR = Path(__file__).parent / "constants"
ELEMENT_LIST_PATH = CONSTANTS_DIR / "element_list.txt"
SCORING_WEIGHTS_PATH = CONSTANTS_DIR / "scoring_weights.json"
RESIDUE_MODIFICATIONS_PATH = CONSTANTS_DIR / "residue_modifications.json"


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


# Load configurations on import
try:
    ATOMIC_MASSES = load_atomic_masses(ELEMENT_LIST_PATH)
except Exception as e:
    print(f"Error loading atomic masses: {e}", file=sys.stderr)
    ATOMIC_MASSES = {}

H_MASS = ATOMIC_MASSES.get("H", 1.007825)
PROTON_MASS = 1.007276
MAX_CHARGE = 2


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
    FragmentationRule("NC", H_MASS, -H_MASS),
    FragmentationRule("OC", H_MASS, -H_MASS),
    FragmentationRule("CC", H_MASS, -H_MASS),
)


@dataclass(frozen=True)
class ChargeWeights:
    """Weights and max rank configuration for a specific precursor charge state."""

    max_rank: int
    charge: int
    missing_peak_weight: float
    weights: tuple[float, ...]


def load_scoring_weights(path: Path) -> dict[int, ChargeWeights]:
    """Parse scoring weights from scoring_weights.json."""
    if not path.exists():
        raise FileNotFoundError(f"Scoring weights configuration file not found at {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    weights_dict = {}
    for item in data:
        charge = item["charge"]
        weights_dict[charge] = ChargeWeights(
            max_rank=item["max_rank"],
            charge=charge,
            missing_peak_weight=item["missing"],
            weights=tuple(item["weights"]),
        )
    return weights_dict


try:
    SCORING_WEIGHTS = load_scoring_weights(SCORING_WEIGHTS_PATH)
except Exception as e:
    print(f"Error loading scoring weights: {e}", file=sys.stderr)
    SCORING_WEIGHTS = {}


def load_residue_modifications(path: Path) -> dict[str, list[dict]]:
    """Parse residue modifications configuration from residue_modifications.json."""
    if not path.exists():
        raise FileNotFoundError(f"Residue modifications file not found at {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


try:
    RESIDUE_MODIFICATIONS = load_residue_modifications(RESIDUE_MODIFICATIONS_PATH)
except Exception as e:
    print(f"Error loading residue modifications: {e}", file=sys.stderr)
    RESIDUE_MODIFICATIONS = {}
