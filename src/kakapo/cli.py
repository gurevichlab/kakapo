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
    DEFAULT_FRAGMENTATION_RULES,
    DEFAULT_MASS_TOLERANCE,
    DEFAULT_MIN_PEAKS,
    DEFAULT_OUTPUT_DIR,
)
from kakapo.io.writer import write_mass_matched_results, write_results
from kakapo.main import run_pipeline
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
        help="Path(s) to rBAN YAML structure files or directories containing YAML files.",
    )

    # Parameters
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=DEFAULT_MASS_TOLERANCE,
        help="Mass tolerance window in Daltons for peak matching.",
    )
    parser.add_argument(
        "--min-peaks",
        type=int,
        default=DEFAULT_MIN_PEAKS,
        help="Minimum number of peaks required to include an experimental spectrum (default: 1).",
    )
    parser.add_argument(
        "--no-filters",
        action="store_false",
        dest="apply_filters",
        default=DEFAULT_APPLY_FILTERS,
        help="Disable applying default matchms filters to experimental spectra.",
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
        choices=["table", "txt", "csv"],
        default="table",
        help="Format of the output results (default: table).",
    )
    parser.add_argument(
        "--mass-match",
        action="store_true",
        help="Output an additional results file where structure mass matches spectrum mass.",
    )
    parser.add_argument(
        "--mass-match-tolerance",
        type=float,
        help="Mass tolerance in Daltons for matching structure mass to spectrum mass.",
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


def determine_output_path(parsed_args: argparse.Namespace) -> Optional[Path | str]:
    """Determine the output file path or stdout option from arguments.

    Args:
        parsed_args: Parsed command-line arguments.

    Returns:
        The output path as a Path, a string (e.g. "-"), or None.
    """
    if parsed_args.stdout:
        return None
    if parsed_args.output == "-":
        return "-"

    base_out = Path(parsed_args.output) if parsed_args.output else DEFAULT_OUTPUT_DIR
    if base_out.is_dir() or not base_out.suffix:
        ext = "csv" if parsed_args.format == "csv" else "txt"
        return base_out / f"results.{ext}"
    return base_out


def determine_matched_mass_output_path(
    output_path: Optional[Path | str],
    format_type: str,
) -> Optional[Path | str]:
    """Determine the output file path for mass-matched results.

    Args:
        output_path: The main output path (Path, "-", or None).
        format_type: Output format ("table", "txt", or "csv").

    Returns:
        The Path to write the mass-matched results, or None / "-" for stdout.
    """
    if output_path is None:
        return None
    if output_path == "-":
        return "-"

    ext = "csv" if format_type.lower() == "csv" else "txt"
    main_path = Path(output_path)
    if main_path.is_dir() or not main_path.suffix:
        return main_path / f"results_mass_matched.{ext}"

    return main_path.with_name(f"{main_path.stem}_mass_matched{main_path.suffix}")


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
    if parsed_args.mass_match and parsed_args.mass_match_tolerance is None:
        parser.error("--mass-match-tolerance is required when --mass-match is specified.")
    if parsed_args.mass_match_tolerance is not None and parsed_args.mass_match_tolerance < 0:
        parser.error("--mass-match-tolerance must be non-negative.")

    # 1. Configure logging
    configure_logging(parsed_args)

    logger.info("Initializing Kakapo CLI")

    # 2. Resolve input paths
    try:
        spectra_files = resolve_paths(parsed_args.spectra, [".mgf"])
        structure_files = resolve_paths(parsed_args.structures, [".yaml", ".yml"])
    except FileNotFoundError as e:
        logger.error("Path validation failed: %s", e)
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not spectra_files:
        logger.error("No valid spectra files (.mgf) resolved from inputs.")
        print("Error: No spectra files (.mgf) resolved from inputs.", file=sys.stderr)
        return 1
    if not structure_files:
        logger.error("No valid structure files (.yaml, .yml) resolved from inputs.")
        print("Error: No structure files (.yaml, .yml) resolved from inputs.", file=sys.stderr)
        return 1

    # 3. Determine output path
    output_path = determine_output_path(parsed_args)

    # 4. Run pipeline
    try:
        results = run_pipeline(
            spectra_files=spectra_files,
            structure_files=structure_files,
            mass_tolerance=parsed_args.tolerance,
            apply_filters=parsed_args.apply_filters,
            min_peaks=parsed_args.min_peaks,
            rules=DEFAULT_FRAGMENTATION_RULES,
        )

        # 5. Write results
        write_results(results, output_path=output_path, format_type=parsed_args.format)

        # 6. Write mass-matched results if requested
        if parsed_args.mass_match:
            matched_output_path = determine_matched_mass_output_path(
                output_path, parsed_args.format
            )
            write_mass_matched_results(
                results,
                output_path=matched_output_path,
                tolerance=parsed_args.mass_match_tolerance,
                format_type=parsed_args.format,
            )

    except Exception as e:
        logger.exception("Pipeline execution failed")
        print(f"Pipeline execution failed: {e}", file=sys.stderr)
        return 1

    total_duration = time.perf_counter() - start_total
    logger.debug("CLI execution completed in %.4f seconds", total_duration)
    logger.info("CLI execution completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
