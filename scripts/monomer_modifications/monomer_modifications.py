"""Script to map chemical modifications to core residues.

This script parses monomer modifications from a TSV monomer table and a JSON monomer database,
calculates chemical formula and mass changes using RDKit, groups them by core residue,
and writes the mapped modifications to a JSON file.
"""

import argparse
import ast
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


class ModificationDetails:
    """Details of a chemical modification.

    Attributes:
        formula_difference: The chemical formula of the change (e.g. +CH2).
        mass_difference: The mass modification in Da.
        modified_monomers: List of names of corresponding modified monomers.
    """

    def __init__(
        self, formula_difference: str, mass_difference: float, modified_monomers: list[str]
    ):
        """Initialize the modification details.

        Args:
            formula_difference: The chemical formula of the change (e.g., +CH2).
            mass_difference: The mass difference in Da.
            modified_monomers: Names of modified monomers contributing this formula.
        """
        self.formula_difference = formula_difference
        self.mass_difference = mass_difference
        self.modified_monomers = sorted(modified_monomers)

    def to_dict(self) -> dict[str, Any]:
        """Convert the modification details to a dictionary.

        Returns:
            A dictionary representation of the modification details.
        """
        return {
            "formula_difference": self.formula_difference,
            "mass_difference": self.mass_difference,
            "modified_monomers": self.modified_monomers,
        }


def parse_formula(formula_str: str) -> dict[str, int]:
    """Parse a chemical formula string into a dictionary of element counts.

    Args:
        formula_str: The chemical formula string (e.g., 'C3H7NO2').

    Returns:
        A dictionary mapping element symbols to their count.
    """
    pattern = re.compile(r"([A-Z][a-z]*)([0-9]*)")
    result: dict[str, int] = {}
    for symbol, count_str in pattern.findall(formula_str):
        count = int(count_str) if count_str else 1
        result[symbol] = result.get(symbol, 0) + count
    return result


def get_formula_difference(core_formula: dict[str, int], mod_formula: dict[str, int]) -> str:
    """Calculate the difference between two chemical formulas.

    Args:
        core_formula: Dictionary of element counts for the core residue.
        mod_formula: Dictionary of element counts for the modified residue.

    Returns:
        A string representing the formula difference (e.g., '+CH2', '-O', '+Cl-H').
    """
    all_elements = set(core_formula.keys()) | set(mod_formula.keys())
    diff: dict[str, int] = {}
    for elem in all_elements:
        d = mod_formula.get(elem, 0) - core_formula.get(elem, 0)
        if d != 0:
            diff[elem] = d

    # Sort elements in Hill system order: C first, H second, then alphabetical
    def sort_key(elem: str) -> tuple[int, str]:
        if elem == "C":
            return (0, "")
        elif elem == "H":
            return (1, "")
        else:
            return (2, elem)

    sorted_elems = sorted(diff.keys(), key=sort_key)

    pos_part = ""
    neg_part = ""

    pos_elems = [e for e in sorted_elems if diff[e] > 0]
    for e in pos_elems:
        count = diff[e]
        pos_part += f"{e}{count if count > 1 else ''}"

    neg_elems = [e for e in sorted_elems if diff[e] < 0]
    for e in neg_elems:
        count = -diff[e]
        neg_part += f"{e}{count if count > 1 else ''}"

    res = ""
    if pos_part:
        res += f"+{pos_part}"
    if neg_part:
        res += f"-{neg_part}"
    return res


def load_element_masses(filepath: Path) -> dict[str, float]:
    """Load chemical element masses from a config file.

    Args:
        filepath: Path to the element list file.

    Returns:
        A dictionary mapping element symbols to their atomic masses.
    """
    masses: dict[str, float] = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                symbol = parts[0]
                try:
                    masses[symbol] = float(parts[1])
                except ValueError:
                    logger.warning("Could not parse mass for element '%s': '%s'", symbol, parts[1])
    return masses


def get_formula_from_smiles(smiles: str) -> dict[str, int]:
    """Deduce chemical formula from a SMILES string using RDKit.

    Args:
        smiles: The SMILES string.

    Returns:
        A dictionary of element counts.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    formula_str = CalcMolFormula(mol)
    return parse_formula(formula_str)


def find_modified_monomer(
    json_data: list[dict[str, Any]], monomer_name: str, nerpa_residue: str
) -> Optional[dict[str, Any]]:
    """Find a modified monomer in the JSON database.

    The monomer is matched by name or code, and the 'monomer' field
    must contain the core residue as a substring.

    Args:
        json_data: List of monomer entries from JSON.
        monomer_name: The name of the modified monomer to search for.
        nerpa_residue: The core residue name.

    Returns:
        The matched monomer entry dict, or None if not found.
    """
    for entry in json_data:
        # Match MonomerName exactly in monomer field or in codes
        if entry.get("monomer") == monomer_name or monomer_name in entry.get("codes", []):
            monomer_field = entry.get("monomer", "")
            if nerpa_residue in monomer_field:
                return entry
    return None


def find_core_monomer(
    json_data: list[dict[str, Any]], nerpa_residue: str
) -> Optional[dict[str, Any]]:
    """Find the core monomer in the JSON database.

    The monomer must match the NerpaResidue exactly by name or code.

    Args:
        json_data: List of monomer entries from JSON.
        nerpa_residue: The core residue name.

    Returns:
        The matched core monomer entry dict, or None if not found.
    """
    for entry in json_data:
        if entry.get("monomer") == nerpa_residue or nerpa_residue in entry.get("codes", []):
            return entry
    return None


def calculate_mass_diff(
    core_formula: dict[str, int], mod_formula: dict[str, int], element_masses: dict[str, float]
) -> float:
    """Calculate the mass difference between two chemical formulas.

    Args:
        core_formula: Dict of element counts for the core residue.
        mod_formula: Dict of element counts for the modified monomer.
        element_masses: Dict mapping element symbols to their masses.

    Returns:
        The calculated mass difference.
    """
    diff_counts = {}
    all_elems = set(core_formula.keys()) | set(mod_formula.keys())
    for elem in all_elems:
        d = mod_formula.get(elem, 0) - core_formula.get(elem, 0)
        if d != 0:
            diff_counts[elem] = d

    mass_diff = 0.0
    for elem, count in diff_counts.items():
        if elem not in element_masses:
            raise ValueError(f"Element {elem} not found in element_list.txt")
        mass_diff += count * element_masses[elem]
    return round(mass_diff, 6)


def resolve_monomer_entries(
    row: dict[str, str], line_num: int, monomers_db: list[dict[str, Any]]
) -> Optional[tuple[dict[str, Any], dict[str, Any]]]:
    """Resolve modified and core monomer entries from the database.

    Args:
        row: A dict representing a TSV row.
        line_num: The line number in the TSV.
        monomers_db: The monomers JSON database.

    Returns:
        A tuple of (modified monomer entry, core monomer entry) or None if error.
    """
    monomer_name = row.get("MonomerName")
    nerpa_residue = row.get("NerpaResidue")
    if not monomer_name or not nerpa_residue:
        logger.warning("Line %d: Missing MonomerName or NerpaResidue", line_num)
        return None

    # Find modified monomer in JSON database
    mod_entry = find_modified_monomer(monomers_db, monomer_name, nerpa_residue)
    if not mod_entry:
        logger.error(
            "Line %d: Modified monomer not found in JSON for MonomerName='%s' "
            "with residue substring '%s'",
            line_num,
            monomer_name,
            nerpa_residue,
        )
        return None

    # Find core monomer in JSON database
    core_entry = find_core_monomer(monomers_db, nerpa_residue)
    if not core_entry:
        logger.error(
            "Line %d: Core monomer not found in JSON for NerpaResidue='%s'",
            line_num,
            nerpa_residue,
        )
        return None

    return mod_entry, core_entry


def process_row(
    row: dict[str, str],
    line_num: int,
    monomers_db: list[dict[str, Any]],
    element_masses: dict[str, float],
    grouped_modifications: dict[str, dict[str, list[str]]],
    mass_cache: dict[str, float],
) -> None:
    """Process a single row from the TSV table and extract modifications.

    Args:
        row: A dict representing a TSV row.
        line_num: The line number in the TSV.
        monomers_db: The monomers JSON database.
        element_masses: Element symbol to mass mapping.
        grouped_modifications: Output mapping structure to update.
        mass_cache: Cache of formula difference to mass change.
    """
    if row.get("Type") != "NRPS" or row.get("NameFormat") != "rBAN/Norine":
        return

    modifications_str = row.get("Modifications", "[]")
    try:
        modifications = ast.literal_eval(modifications_str)
    except (ValueError, SyntaxError) as e:
        logger.warning(
            "Line %d: Could not parse modifications list '%s': %s",
            line_num,
            modifications_str,
            e,
        )
        return

    if (
        not isinstance(modifications, list)
        or not modifications
        or modifications == ["E"]
        or modifications == ["MT"]
    ):
        return

    res = resolve_monomer_entries(row, line_num, monomers_db)
    if not res:
        return
    mod_entry, core_entry = res

    monomer_name = row["MonomerName"]
    nerpa_residue = row["NerpaResidue"]

    mod_smiles = mod_entry.get("smiles")
    core_smiles = core_entry.get("smiles")
    if not mod_smiles or not core_smiles:
        logger.error(
            "Line %d: Missing SMILES for modified '%s' (%s) or core '%s' (%s)",
            line_num,
            monomer_name,
            mod_smiles,
            nerpa_residue,
            core_smiles,
        )
        return

    try:
        mod_formula = get_formula_from_smiles(mod_smiles)
        core_formula = get_formula_from_smiles(core_smiles)
    except ValueError as e:
        logger.error("Line %d: Failed to parse SMILES or calculate formula: %s", line_num, e)
        return

    formula_diff = get_formula_difference(core_formula, mod_formula)
    if not formula_diff:
        return

    # Calculate mass difference
    if formula_diff not in mass_cache:
        try:
            mass_cache[formula_diff] = calculate_mass_diff(
                core_formula, mod_formula, element_masses
            )
        except ValueError as e:
            logger.error("Line %d: Mass calculation failed: %s", line_num, e)
            return

    # Group the modifications
    grouped_modifications.setdefault(nerpa_residue, {})
    residue_group = grouped_modifications[nerpa_residue]
    residue_group.setdefault(formula_diff, []).append(monomer_name)


def main() -> None:
    """Main execution function to perform mapping of monomer modifications."""
    parser = argparse.ArgumentParser(description="Map monomer modifications to core residues.")
    parser.add_argument(
        "--tsv",
        type=Path,
        default=Path(__file__).parent / "monomer_names_table.tsv",
        help="Path to monomer names TSV file",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).parent / "nrproMonomers.json",
        help="Path to nrproMonomers JSON file",
    )
    parser.add_argument(
        "--elements",
        type=Path,
        default=Path(__file__).parent / "element_list.txt",
        help="Path to element list file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "residue_modifications.json",
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    # Load element masses
    if not args.elements.exists():
        logger.error("Elements file not found: %s", args.elements)
        sys.exit(1)
    element_masses = load_element_masses(args.elements)

    # Load monomers database
    if not args.json.exists():
        logger.error("Monomers JSON database file not found: %s", args.json)
        sys.exit(1)
    with open(args.json, encoding="utf-8") as f:
        try:
            monomers_db = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON database %s: %s", args.json, e)
            sys.exit(1)

    # Dictionary to collect results: core_residue -> formula_diff -> list of modified_monomers
    grouped_modifications: dict[str, dict[str, list[str]]] = {}
    # Dictionary to cache calculated mass differences: formula_diff -> mass_diff
    mass_cache: dict[str, float] = {}

    # Read TSV table
    if not args.tsv.exists():
        logger.error("TSV monomer names table not found: %s", args.tsv)
        sys.exit(1)

    with open(args.tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for line_num, row in enumerate(reader, start=2):
            process_row(
                row, line_num, monomers_db, element_masses, grouped_modifications, mass_cache
            )

    # Convert the grouped modifications to output format using the class
    output_data: dict[str, list[dict[str, Any]]] = {}
    for core_residue in sorted(grouped_modifications.keys()):
        residue_mods = grouped_modifications[core_residue]
        mod_details_list = []
        for formula_diff in sorted(residue_mods.keys()):
            monomer_names = list(set(residue_mods[formula_diff]))
            mass_diff = mass_cache[formula_diff]
            detail = ModificationDetails(formula_diff, mass_diff, monomer_names)
            mod_details_list.append(detail.to_dict())
        output_data[core_residue] = mod_details_list

    # Write output to JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Successfully wrote mapping to %s", args.output)


if __name__ == "__main__":
    main()
