"""Main pipeline orchestration module for kakapo."""

from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

from kakapo.constants import DEFAULT_RULES, FragmentationRule
from kakapo.core.fragmenter import generate_theoretical_spectrum
from kakapo.core.matcher import match_spectra
from kakapo.io.mgf_parser import load_mgf_batch
from kakapo.io.yaml_parser import parse_rban_yaml_batch
from kakapo.models import MatchResult
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(
    spectra_files: Union[str, Path, Iterable[Union[str, Path]]],
    structure_files: Union[str, Path, Iterable[Union[str, Path]]],
    mass_tolerance: float,
    apply_filters: bool = True,
    min_peaks: int = 1,
    rules: Optional[tuple[FragmentationRule, ...]] = None,
) -> list[MatchResult]:
    """Run the main kakapo analysis pipeline.

    Coordinates input loading, theoretical spectrum generation (fragmentation), spectrum matching,
    and sorting of the match results.

    Args:
        spectra_files: Single path or list of paths to MGF experimental spectra files.
        structure_files: Single path or list of paths to rBAN YAML structure files.
        mass_tolerance: The maximum absolute mass difference (in Da) allowed for matching peaks.
        apply_filters: Whether to clean up experimental spectra using matchms filters.
        min_peaks: Minimum number of peaks required to include an experimental spectrum.
        rules: Fragmentation rules to apply. Defaults to DEFAULT_RULES if None.

    Returns:
        A sorted list of MatchResult objects. Results are sorted primarily by matched_peaks_count in
        descending order, then secondarily by spectrum_id (ascending), and thirdly by structure_id
        (ascending).

    Raises:
        ValueError: If inputs are invalid or no structures/spectra are found.
        FileNotFoundError: If any of the input files do not exist.
    """
    logger.info("Starting Kakapo workflow pipeline")

    # Normalize spectra files to a list of Path objects
    if isinstance(spectra_files, (str, Path)):
        spec_paths = [Path(spectra_files)]
    else:
        spec_paths = [Path(p) for p in spectra_files]

    # Normalize structure files to a list of Path objects
    if isinstance(structure_files, (str, Path)):
        struct_paths = [Path(structure_files)]
    else:
        struct_paths = [Path(p) for p in structure_files]

    logger.debug(
        "Normalized paths: %d spectra file(s), %d structure file(s)",
        len(spec_paths),
        len(struct_paths),
    )

    if not spec_paths:
        logger.error("No spectra files provided to run_pipeline")
        raise ValueError("No spectra files provided")
    if not struct_paths:
        logger.error("No structure files provided to run_pipeline")
        raise ValueError("No structure files provided")

    # 1. Load experimental spectra
    logger.info("Loading experimental spectra from %d file(s)", len(spec_paths))
    spectra = load_mgf_batch(
        file_paths=spec_paths,  # pyright: ignore [reportArgumentType] (paths are normalized)
        apply_filters=apply_filters,
        min_peaks=min_peaks,
    )
    logger.info("Successfully loaded %d experimental spectra", len(spectra))

    # 2. Load and parse structures
    logger.info("Loading structures from %d file(s)", len(struct_paths))
    molecules = parse_rban_yaml_batch(struct_paths)
    logger.info("Successfully loaded %d structures", len(molecules))

    # Use default fragmentation rules if none provided
    frag_rules = rules if rules is not None else DEFAULT_RULES

    # 3. Generate theoretical spectra
    logger.info("Generating theoretical spectra for %d structures", len(molecules))
    theoretical_spectra = []
    for mol in molecules.values():
        logger.debug("Generating theoretical spectrum for structure: %s", mol.structure_id)
        theo_spec = generate_theoretical_spectrum(mol, rules=frag_rules)
        theoretical_spectra.append(theo_spec)

    # 4. Pairwise spectrum matching
    logger.info("Matching %d spectra against %d structures", len(spectra), len(theoretical_spectra))
    results = match_spectra(
        spectra=spectra,
        theoretical_spectra=theoretical_spectra,
        mass_tolerance=mass_tolerance,
    )

    # 5. Sort results
    # Primary key: matched_peaks_count (descending)
    # Secondary key: spectrum_id (ascending)
    # Tertiary key: structure_id (ascending)
    logger.info("Sorting %d match results", len(results))
    results.sort(key=lambda r: (-r.matched_peaks_count, r.spectrum_id, r.structure_id))

    logger.info("Pipeline execution complete")
    return results
