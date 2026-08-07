"""Command-line interface for kakapo.

This module provides the command-line entry point to parse arguments,
set up logging, invoke the orchestration pipeline, and write output results.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from kakapo.config import (
    DEFAULT_APPLY_FILTERS,
    DEFAULT_CACHE_DIR,
    DEFAULT_FRAGMENTATION_RULES,
    DEFAULT_MASS_TOLERANCE,
    DEFAULT_MIN_PEAKS,
    DEFAULT_MIN_TOLERANCE,
    DEFAULT_MODIFICATION_MODE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SCORING_MODE,
    DEFAULT_USE_CACHE,
)
from kakapo.io.writer import write_results, write_spectra, write_structures
from kakapo.main import run_pipeline
from kakapo.models import ModificationMode, ScoringMode, Tolerance
from kakapo.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def resolve_paths(paths: list[str], extensions: list[str]) -> list[Path]:
    """Resolve a list of paths (files or directories) to a flat list of file paths.

    If a path is a directory, searches for files with the given extensions.
    Otherwise, treats it as a file path.

    Args:
        paths: List of file or directory paths as strings.
        extensions: List of file extensions to search for in directories (e.g. ['.mgf']).

    Returns:
        A list of resolved Path objects.

    Raises:
        FileNotFoundError: If any of the input paths do not exist.
    """
    resolved: list[Path] = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: '{path_str}'")

        if path.is_dir():
            found_any = False
            for ext in extensions:
                pattern_lower = f"*{ext.lower()}"
                pattern_upper = f"*{ext.upper()}"
                for file_path in path.glob(pattern_lower):
                    if file_path.is_file():
                        resolved.append(file_path)
                        found_any = True
                for file_path in path.glob(pattern_upper):
                    if file_path.is_file() and file_path not in resolved:
                        resolved.append(file_path)
                        found_any = True
            if not found_any:
                logger.warning(
                    "No files with extension(s) %s found in directory '%s'",
                    extensions,
                    path_str,
                )
        else:
            resolved.append(path)
    return resolved


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for kakapo.

    Returns:
        An ArgumentParser instance populated with all options and help messages.
    """
    parser = argparse.ArgumentParser(
        description="Kakapo: Spectrum matching tool for mass spectrometry data analysis."
    )

    # Input options
    parser.add_argument(
        "--spectra",
        "-s",
        nargs="+",
        required=True,
        help="Path(s) to experimental MGF spectra files or directories containing MGF files.",
    )
    parser.add_argument(
        "--structures",
        "--structs",
        "-str",
        nargs="+",
        required=True,
        help="Path(s) to BGC variant JSON structure files or directories containing them.",
    )

    # Parameters
    parser.add_argument(
        "--tolerance",
        "-t",
        type=str,
        default=DEFAULT_MASS_TOLERANCE,
        help="Mass tolerance window (e.g. '20ppm', '0.02da', or '0.02' for Da).",
    )
    parser.add_argument(
        "--min-tolerance",
        type=float,
        default=DEFAULT_MIN_TOLERANCE,
        help="Minimum absolute mass tolerance in Daltons (only applied for ppm).",
    )
    parser.add_argument(
        "--min-peaks",
        type=int,
        default=DEFAULT_MIN_PEAKS,
        help="Minimum number of peaks required to include an experimental spectrum (default: 5).",
    )
    parser.add_argument(
        "--no-filters",
        action="store_false",
        dest="apply_filters",
        default=DEFAULT_APPLY_FILTERS,
        help="Disable applying default matchms filters to experimental spectra.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        default=DEFAULT_USE_CACHE,
        help="Disable caching of parsed/cleaned experimental spectra.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Directory where parsed/cleaned experimental spectra are cached.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=[m.value for m in ModificationMode],
        default=DEFAULT_MODIFICATION_MODE,
        help=(
            "Modification matching mode (default: free-mod). "
            "Choices: no-mod (unmodified only), "
            "free-mod (single modification on any monomer), "
            "x-mod (single modification only on X monomers), "
            "two-mod (known modification on base monomer + residual on X monomer)."
        ),
    )
    parser.add_argument(
        "--scoring",
        choices=[m.value for m in ScoringMode],
        default=DEFAULT_SCORING_MODE,
        help="Scoring mode to use for matching (default: all-peaks). "
        "Choices: all-peaks, exp-peaks, intencity.",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help=(
            "Path to output file or directory. If a directory or not provided, "
            "saves to the default output directory. If '-' is provided, prints to stdout."
        ),
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Force printing results to stdout (takes precedence over --output).",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["table", "txt", "csv", "json"],
        nargs="+",
        default=["table"],
        help="One or more output formats (default: table). Choices: table, txt, csv, json.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        help="Output only top N match results by number of matched peaks.",
    )

    # Logging options
    parser.add_argument(
        "--log-file",
        "-l",
        type=str,
        help="Path to write log file. Defaults to a log file in the default log directory.",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Disable writing logs to a file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (can be specified multiple times, e.g. -v, -vv).",
    )

    return parser


def configure_logging(parsed_args: argparse.Namespace) -> None:
    """Configure logging based on CLI arguments.

    Determines the log file path and calls setup_logging.

    Args:
        parsed_args: Parsed command-line arguments.
    """
    if parsed_args.no_log_file:
        log_path = None
    elif parsed_args.log_file:
        log_path = Path(parsed_args.log_file)
    else:
        if parsed_args.output and parsed_args.output != "-":
            base_out = Path(parsed_args.output)
            if base_out.is_dir() or not base_out.suffix:
                output_dir = base_out
            else:
                output_dir = base_out.parent
        else:
            output_dir = DEFAULT_OUTPUT_DIR
        log_path = output_dir / "kakapo.log"

    verbosity = min(parsed_args.verbose, 2)
    setup_logging(verbosity=verbosity, log_file=log_path)


def determine_output_path(
    parsed_args: argparse.Namespace,
    format_type: str,
) -> Optional[Path | str]:
    """Determine the output file path or stdout option from arguments for a single format.

    Args:
        parsed_args: Parsed command-line arguments.
        format_type: The output format for which to determine the path.

    Returns:
        The output path as a Path, a string (e.g. "-"), or None.
    """
    if parsed_args.stdout:
        return None
    if parsed_args.output == "-":
        return "-"

    base_out = Path(parsed_args.output) if parsed_args.output else DEFAULT_OUTPUT_DIR
    if base_out.is_dir() or not base_out.suffix:
        fmt = format_type.lower()
        ext = fmt if fmt in ("csv", "json") else "txt"
        return base_out / f"results.{ext}"
    # If a specific file path was provided and there are multiple formats, use it for the
    # first format only; subsequent formats fall back to the same directory with auto names.
    return base_out


def _write_all_formats(
    results: list,
    molecules: list,
    spectra: list,
    parsed_args: argparse.Namespace,
) -> None:
    """Write results for every requested output format.

    Args:
        results: Matched results to write.
        molecules: Loaded structures to filter and write.
        spectra: Loaded spectra to filter and write.
        parsed_args: Parsed command-line arguments (provides format list, output path, etc.).
    """
    # Filter only used molecules by ids in results
    matched_ids = {r.structure_id for r in results}
    used_molecules = [mol for mol in molecules if mol.structure_id in matched_ids]

    # Filter only used spectra by ids in results
    matched_spec_ids = {r.spectrum_id for r in results}
    used_spectra = [spec for spec in spectra if spec.spectrum_id in matched_spec_ids]

    structures_written = False
    spectra_written = False
    for fmt in parsed_args.format:
        output_path = determine_output_path(parsed_args, fmt)
        write_results(results, output_path=output_path, format_type=fmt, molecules=used_molecules)
        if not structures_written and output_path is not None and str(output_path) != "-":
            write_structures(used_molecules, output_path=output_path)
            structures_written = True
        if not spectra_written and output_path is not None and str(output_path) != "-":
            write_spectra(used_spectra, output_path=output_path)
            spectra_written = True


def main(args: Optional[list[str]] = None) -> int:
    """Command-line interface entry point for kakapo.

    Parses command-line arguments, validates inputs, configures logging,
    runs the spectrum matching pipeline, and outputs results.

    Args:
        args: List of command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    start_total = time.perf_counter()
    parser = create_parser()
    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Validate arguments
    if parsed_args.top_n is not None and parsed_args.top_n <= 0:
        parser.error("--top-n must be positive.")

    try:
        tolerance = Tolerance.from_string(
            parsed_args.tolerance, min_tolerance=parsed_args.min_tolerance
        )
    except ValueError as e:
        parser.error(str(e))

    # 1. Configure logging
    configure_logging(parsed_args)

    logger.info("Initializing Kakapo CLI")

    # 2. Resolve input paths
    try:
        spectra_files = resolve_paths(parsed_args.spectra, [".mgf"])
        structure_files = resolve_paths(parsed_args.structures, [".json"])
    except FileNotFoundError as e:
        logger.error("Path validation failed: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not spectra_files:
        logger.error("No valid spectra files (.mgf) resolved from inputs.")
        print("Error: No spectra files (.mgf) resolved from inputs.", file=sys.stderr)
        return 1
    if not structure_files:
        logger.error("No valid structure files (.json) resolved from inputs.")
        print(
            "Error: No structure files (.json) resolved from inputs.",
            file=sys.stderr,
        )
        return 1

    # 3. Run pipeline
    try:
        results, molecules, spectra = run_pipeline(
            spectra_files=spectra_files,
            structure_files=structure_files,
            tolerance=tolerance,
            apply_filters=parsed_args.apply_filters,
            min_peaks=parsed_args.min_peaks,
            rules=DEFAULT_FRAGMENTATION_RULES,
            use_cache=parsed_args.use_cache,
            cache_dir=Path(parsed_args.cache_dir),
            mode=ModificationMode(parsed_args.mode),
            scoring=ScoringMode(parsed_args.scoring),
        )

        # 5. Write results (one file per requested format)
        if parsed_args.top_n is not None:
            results = results[: parsed_args.top_n]
        _write_all_formats(results, molecules, spectra, parsed_args)

    except Exception as e:
        logger.exception("Pipeline execution failed")
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        return 1

    total_duration = time.perf_counter() - start_total
    logger.info("CLI execution completed in %.4f seconds", total_duration)
    logger.info("CLI execution completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
