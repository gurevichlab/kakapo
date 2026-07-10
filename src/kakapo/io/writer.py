"""Output formatting and writing module for spectrum match results.

Supports exporting results as standard CSV strings or pretty-printed ASCII tables
to files or standard output.
"""

import csv
import io
import json
import sys
from pathlib import Path
from typing import Optional, Union

from tabulate import tabulate

from kakapo.io.yaml_parser import _serialize_molecule
from kakapo.models import MatchResult, Molecule
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def format_as_csv(results: list[MatchResult]) -> str:
    """Format a list of MatchResult objects into a standard CSV string.

    Args:
        results: A list of MatchResult instances to format.

    Returns:
        A standard RFC 4180 compliant CSV string representation.
    """
    logger.debug("Formatting %d match results as CSV", len(results))
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    # Header columns matching MatchResult attributes
    headers = [
        "structure_id",
        "spectrum_id",
        "structure_mass",
        "spectrum_mass",
        "mass_tolerance",
        "matched_peaks_count",
        "theoretical_peaks_count",
        "experimental_peaks_count",
        "monomer_count",
        "modified_monomer_idx",
        "mass_mod",
        "initial_modified_monomer_mass",
    ]
    writer.writerow(headers)

    for result in results:
        writer.writerow(
            [
                result.structure_id,
                result.spectrum_id,
                result.structure_mass,
                result.spectrum_mass,
                result.mass_tolerance,
                result.matched_peaks_count,
                result.theoretical_peaks_count,
                result.experimental_peaks_count,
                result.monomer_count,
                result.modified_monomer_idx,
                result.mass_mod,
                result.initial_modified_monomer_mass,
            ]
        )

    return output.getvalue()


def format_as_table(results: list[MatchResult]) -> str:
    """Format a list of MatchResult objects into a pretty-printed ASCII table.

    Args:
        results: A list of MatchResult instances to format.

    Returns:
        A string containing the pretty-printed table.
    """
    logger.debug("Formatting %d match results as ASCII table", len(results))

    headers = [
        "Structure ID",
        "Spectrum ID",
        "Structure Mass",
        "Spectrum Mass",
        "Tolerance",
        "Matched Peaks",
        "Theoretical Peaks",
        "Experimental Peaks",
        "Monomer Count",
        "Modified Monomer Index",
        "Mass Mod",
        "Initial Modified Monomer Mass",
    ]

    data = [
        [
            result.structure_id,
            result.spectrum_id,
            result.structure_mass,
            result.spectrum_mass,
            result.mass_tolerance,
            result.matched_peaks_count,
            result.theoretical_peaks_count,
            result.experimental_peaks_count,
            result.monomer_count,
            result.modified_monomer_idx,
            result.mass_mod,
            result.initial_modified_monomer_mass,
        ]
        for result in results
    ]

    return tabulate(data, headers=headers, tablefmt="simple")


def mask_to_reversed_binary_string(mask: int, length: int) -> str:
    """Convert a bitmask to a reversed binary string matching the mass vector order.

    Example: mask = 5 (binary 101, monomers 0 and 2 active), length = 4
    returns: "1010"
    """
    chars = []
    for i in range(length):
        if mask & (1 << i):
            chars.append("1")
        else:
            chars.append("0")
    return "".join(chars)


def format_as_json(results: list[MatchResult]) -> str:
    """Format a list of MatchResult objects into a standard JSON string.

    Args:
        results: A list of MatchResult instances to format.

    Returns:
        A standard JSON string representation with 2-space indentation.
    """
    logger.debug("Formatting %d match results as JSON", len(results))
    serialized = []
    for r in results:
        cleaned_spec_data = {
            "spectrum_id": r.cleaned_spectrum.spectrum_id,
            "source_file_path": r.cleaned_spectrum.source_file_path,
            "peaks": [
                {"mass": p.mass, "intensity": p.intensity}
                for p in r.cleaned_spectrum.get_sorted_peaks()
            ],
            "precursor_mass": r.cleaned_spectrum.precursor_mass,
            "charge": r.cleaned_spectrum.charge,
        }

        matched_peaks_data = []
        for mp in r.matched_peaks:
            mask_str = mask_to_reversed_binary_string(mp.theoretical_fragment_mask, r.monomer_count)
            matched_peaks_data.append(
                {
                    "theoretical_mz": mp.theoretical_mz,
                    "experimental_mz": mp.experimental_mz,
                    "charge": mp.charge,
                    "theoretical_fragment_mask": mask_str,
                }
            )

        serialized.append(
            {
                "structure_id": r.structure_id,
                "spectrum_id": r.spectrum_id,
                "cleaned_spectrum": cleaned_spec_data,
                "structure_mass": r.structure_mass,
                "spectrum_mass": r.spectrum_mass,
                "mass_tolerance": r.mass_tolerance,
                "matched_peaks_count": r.matched_peaks_count,
                "theoretical_peaks_count": r.theoretical_peaks_count,
                "experimental_peaks_count": r.experimental_peaks_count,
                "monomer_count": r.monomer_count,
                "unmodified_monomer_mass_vector": list(r.unmodified_monomer_mass_vector),
                "matched_peaks": matched_peaks_data,
                "modified_monomer_idx": r.modified_monomer_idx,
                "mass_mod": r.mass_mod,
                "initial_modified_monomer_mass": r.initial_modified_monomer_mass,
            }
        )

    return json.dumps(serialized, indent=2)


def write_results(
    results: list[MatchResult],
    output_path: Optional[Union[str, Path]] = None,
    format_type: str = "table",
) -> None:
    """Write formatted match results to a file or stdout.

    If output_path is None or "-", outputs to stdout. Otherwise, writes to the
    given file path. If the parent directories do not exist, they are created automatically.

    Args:
        results: A list of MatchResult instances.
        output_path: Path to write results. If None or "-", prints to standard output.
        format_type: Output format ("table", "txt", "csv", or "json"). Case-insensitive.

    Raises:
        ValueError: If an unsupported format_type is provided.
        OSError: If writing to the file fails.
    """
    logger.info(
        "Writing %d results to %s with format '%s'",
        len(results),
        output_path if output_path is not None else "stdout",
        format_type,
    )

    fmt = format_type.lower()
    if fmt in ("table", "txt"):
        formatted_content = format_as_table(results)
    elif fmt == "csv":
        formatted_content = format_as_csv(results)
    elif fmt == "json":
        formatted_content = format_as_json(results)
    else:
        logger.error("Unsupported output format type: %s", format_type)
        raise ValueError(
            f"Unsupported format type: {format_type}. Supported formats are: 'table', 'txt', 'csv',"
            " 'json'."
        )

    if output_path is None or str(output_path) == "-":
        logger.debug("Writing to stdout")
        sys.stdout.write(formatted_content)
        if not formatted_content.endswith("\n"):
            sys.stdout.write("\n")
    else:
        file_path = Path(output_path)
        logger.debug("Writing to file path: %s", file_path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(formatted_content, encoding="utf-8")
            logger.info("Successfully wrote results to %s", file_path)
        except OSError as e:
            logger.error("Failed to write results to %s: %s", file_path, e)
            raise


def write_structures(
    molecules: list[Molecule],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Write serialized molecules to a file in JSON format.

    If output_path is None, or stdout/stderr/'-', this does not write anything
    to avoid polluting the results stream. If it points to a file, writes to that file.
    If it points to a directory (or has no suffix), writes to 'structures.json' in
    that directory. If output_path is a specific file, writes to 'structures.json' in
    the same parent directory.

    Args:
        molecules: A list of Molecule instances to serialize.
        output_path: Target file path or directory.
    """
    if output_path is None or str(output_path) == "-":
        logger.debug("Stdout output requested, skipping writing structures JSON")
        return

    path = Path(output_path)
    if path.is_dir() or not path.suffix:
        structures_path = path / "structures.json"
    else:
        structures_path = path.parent / "structures.json"

    logger.info("Writing %d structures to %s", len(molecules), structures_path)

    serialized_data = {mol.structure_id: _serialize_molecule(mol) for mol in molecules}

    try:
        structures_path.parent.mkdir(parents=True, exist_ok=True)
        structures_path.write_text(json.dumps(serialized_data, indent=2), encoding="utf-8")
        logger.info("Successfully wrote structures to %s", structures_path)
    except OSError as e:
        logger.error("Failed to write structures to %s: %s", structures_path, e)
        raise
