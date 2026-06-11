"""Output formatting and writing module for spectrum match results.

Supports exporting results as standard CSV strings or pretty-printed ASCII tables
to files or standard output.
"""

import csv
import io
import sys
from pathlib import Path
from typing import Optional, Union

from tabulate import tabulate

from kakapo.models import MatchResult
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
        ]
        for result in results
    ]

    return tabulate(data, headers=headers, tablefmt="simple")


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
        format_type: Output format ("table", "txt", or "csv"). Case-insensitive.

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
    else:
        logger.error("Unsupported output format type: %s", format_type)
        raise ValueError(
            f"Unsupported format type: {format_type}. Supported formats are: 'table', 'txt', 'csv'."
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
