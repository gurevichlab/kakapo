"""Main pipeline orchestration module for kakapo."""

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

from kakapo.config import DEFAULT_MODIFICATION_MODE, DEFAULT_SCORING_MODE
from kakapo.constants import DEFAULT_RULES, FragmentationRule
from kakapo.core.fragmenter import generate_theoretical_spectrum
from kakapo.core.matcher import match_spectra
from kakapo.io.json_parser import parse_nerpa_json_batch
from kakapo.io.mgf_parser import load_mgf_batch
from kakapo.models import MatchResult, ModificationMode, Molecule, ScoringMode, Spectrum, Tolerance
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def run_pipeline(  # noqa: C901
    spectra_files: Union[str, Path, Iterable[Union[str, Path]]],
    structure_files: Union[str, Path, Iterable[Union[str, Path]]],
    tolerance: Tolerance,
    apply_filters: bool,
    min_peaks: int,
    rules: Optional[tuple[FragmentationRule, ...]],
    use_cache: bool,
    cache_dir: Path,
    mode: ModificationMode = ModificationMode(DEFAULT_MODIFICATION_MODE),  # noqa: B008
    scoring: ScoringMode = ScoringMode(DEFAULT_SCORING_MODE),  # noqa: B008
) -> tuple[list[MatchResult], list[Molecule], list[Spectrum]]:
    """Run the main kakapo analysis pipeline.

    Coordinates input loading, theoretical spectrum generation (fragmentation), spectrum matching,
    and sorting of the match results.

    Args:
        spectra_files: Single path or list of paths to MGF experimental spectra files.
        structure_files: Single path or list of paths to nerpa structure files.
        tolerance: The Tolerance configuration allowed for matching peaks.
        apply_filters: Whether to clean up experimental spectra using matchms filters.
        min_peaks: Minimum number of peaks required to include an experimental spectrum.
        rules: Fragmentation rules to apply.
        use_cache: Whether to use caching.
        cache_dir: Path to cache directory.
        mode: ModificationMode configuration.
        scoring: Scoring mode.

    Returns:
        A tuple of:
            - A sorted list of MatchResult objects. Results are sorted primarily by
              matched_peaks_count in descending order, then secondarily by spectrum_id
              (ascending), and thirdly by structure_id (ascending).
            - A list of loaded Molecule objects.
            - A list of loaded experimental Spectrum objects.

    Raises:
        ValueError: If inputs are invalid or no structures/spectra are found.
        FileNotFoundError: If any of the input files do not exist.
    """
    logger.info("Starting Kakapo workflow pipeline")
    logger.info(
        "Running pipeline with parameters: spectra_files=%s, structure_files=%s, "
        "tolerance=%s, apply_filters=%s, min_peaks=%s, rules=%s, use_cache=%s, cache_dir=%s, "
        "mode=%s, scoring=%s",
        spectra_files,
        structure_files,
        tolerance,
        apply_filters,
        min_peaks,
        rules,
        use_cache,
        cache_dir,
        mode,
        scoring,
    )

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

    if not spec_paths:
        logger.error("No spectra files provided to run_pipeline")
        raise ValueError("No spectra files provided")
    if not struct_paths:
        logger.error("No structure files provided to run_pipeline")
        raise ValueError("No structure files provided")

    for p in spec_paths:
        if not p.exists():
            logger.error("Spectra file not found: %s", p)
            raise FileNotFoundError(f"Spectra file not found: {p}")
    for p in struct_paths:
        if not p.exists():
            logger.error("Structure file not found: %s", p)
            raise FileNotFoundError(f"Structure file not found: {p}")

    logger.debug(
        "Normalized paths: %d spectra file(s), %d structure file(s)",
        len(spec_paths),
        len(struct_paths),
    )

    # 1. Load experimental spectra
    logger.info("Loading experimental spectra from %d file(s)", len(spec_paths))
    start_load_spectra = time.perf_counter()
    spectra = load_mgf_batch(
        file_paths=spec_paths,  # pyright: ignore [reportArgumentType] (paths are normalized)
        cache_dir=cache_dir,
        apply_filters=apply_filters,
        min_peaks=min_peaks,
        use_cache=use_cache,
    )
    duration_load_spectra = time.perf_counter() - start_load_spectra
    logger.info(
        "Loaded %d experimental spectra in %.4f seconds",
        len(spectra),
        duration_load_spectra,
    )
    logger.info("Successfully loaded %d experimental spectra", len(spectra))

    # 2. Load and parse structures
    logger.info("Loading structures from %d file(s)", len(struct_paths))
    start_load_structures = time.perf_counter()

    json_paths = [p for p in struct_paths if p.suffix.lower() == ".json"]
    if not json_paths:
        logger.error("No valid JSON structure files found among provided paths")
        raise ValueError("No valid JSON structure files provided")

    molecules = parse_nerpa_json_batch(json_paths, cache_dir=cache_dir, use_cache=use_cache)

    duration_load_structures = time.perf_counter() - start_load_structures
    logger.info(
        "Loaded %d structures in %.4f seconds",
        len(molecules),
        duration_load_structures,
    )
    logger.info("Successfully loaded %d structures", len(molecules))

    # Use default fragmentation rules if none provided
    frag_rules = rules if rules is not None else DEFAULT_RULES

    # 3. Generate theoretical spectra
    logger.info("Generating theoretical spectra for %d structures", len(molecules))
    start_gen_spectra = time.perf_counter()
    theoretical_spectra = []
    for mol in molecules.values():
        logger.debug("Generating theoretical spectrum for structure: %s", mol.structure_id)
        theo_spec = generate_theoretical_spectrum(mol, rules=frag_rules)
        theoretical_spectra.append(theo_spec)
    duration_gen_spectra = time.perf_counter() - start_gen_spectra
    logger.info(
        "Generated %d theoretical spectra in %.4f seconds",
        len(theoretical_spectra),
        duration_gen_spectra,
    )

    # 4. Pairwise spectrum matching
    logger.info("Matching %d spectra against %d structures", len(spectra), len(theoretical_spectra))
    start_match = time.perf_counter()
    results = match_spectra(
        spectra=spectra,
        theoretical_spectra=theoretical_spectra,
        tolerance=tolerance,
        min_shared_peaks=min_peaks,
        mode=mode,
        scoring=scoring,
    )

    duration_match = time.perf_counter() - start_match
    logger.info(
        "Matched spectra in %.4f seconds",
        duration_match,
    )

    # 5. Sort results
    # Primary key: score (descending)
    # Secondary key: spectrum_id (ascending)
    # Tertiary key: structure_id (ascending)
    logger.info("Sorting %d match results by score", len(results))
    start_sort = time.perf_counter()
    results.sort(key=lambda r: (-r.score, r.spectrum_id, r.structure_id))
    duration_sort = time.perf_counter() - start_sort
    logger.info(
        "Sorted match results in %.4f seconds",
        duration_sort,
    )

    logger.info("Pipeline execution complete")
    return results, list(molecules.values()), spectra
