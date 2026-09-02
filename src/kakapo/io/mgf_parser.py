"""MGF (Mascot Generic Format) parser for mass spectrometry data.

This module handles:
- Loading .mgf files using native_mgf_parser in parallel
- Filtering and cleaning spectra using native NumPy filters
- Caching parsed spectra in JSON format
"""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from kakapo.io.native_mgf_parser import parse_mgf_file
from kakapo.models import Spectrum
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def _get_cache_path(
    file_path: Path,
    apply_filters: bool,
    min_peaks: int,
    cache_dir: Path,
) -> Path:
    """Generate a unique cache path based on the file path, modification time, and parameters.

    Args:
        file_path: Path to the original MGF file.
        apply_filters: Whether filters were applied.
        min_peaks: Minimum peaks filter value.
        cache_dir: Path to the cache directory.

    Returns:
        Path to the JSON cache file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use modification time and file size to automatically invalidate cache if original file changes
    stat_info = file_path.stat()
    mtime = stat_info.st_mtime
    size = stat_info.st_size

    # Build unique hash based on path, mtime, size, and configuration params
    hash_input = f"{file_path.resolve()}_{mtime}_{size}_{apply_filters}_{min_peaks}"
    file_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    return cache_dir / f"{file_path.name}_{file_hash}.json"


def _load_from_cache(cache_path: Path) -> list[Spectrum]:
    """Load serialized spectra from a JSON cache file.

    Args:
        cache_path: Path to the JSON cache file.

    Returns:
        List of parsed kakapo Spectrum objects.
    """
    logger.debug("Loading spectra from cache file: %s", cache_path)
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)

    spectra = []
    for item in data:
        mzs = np.array(item["mzs"], dtype=np.float64)
        intensities = np.array(item["intensities"], dtype=np.float64)

        spectra.append(
            Spectrum(
                spectrum_id=item["spectrum_id"],
                source_file_path=item["source_file_path"],
                mzs=mzs,
                intensities=intensities,
                precursor_mass=item["precursor_mass"],
                charge=item["charge"],
            )
        )
    return spectra


def _save_to_cache(spectra: list[Spectrum], cache_path: Path) -> None:
    """Save parsed spectra to a JSON cache file.

    Args:
        spectra: List of parsed kakapo Spectrum objects.
        cache_path: Path to save the JSON cache file.
    """
    logger.debug("Saving %d spectra to cache file: %s", len(spectra), cache_path)
    data = [
        {
            "spectrum_id": spec.spectrum_id,
            "precursor_mass": spec.precursor_mass,
            "charge": spec.charge,
            "mzs": spec.mzs.tolist(),
            "intensities": spec.intensities.tolist(),
            "source_file_path": spec.source_file_path,
        }
        for spec in spectra
    ]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _ensure_unique_spectrum_ids(spectra: list[Spectrum]) -> list[Spectrum]:
    """Ensure all spectra in list have unique IDs."""
    initial_ids = [spec.spectrum_id for spec in spectra]
    id_counts = Counter(initial_ids)

    unique_spectra = []
    for i, spec in enumerate(spectra):
        init_id = spec.spectrum_id
        if id_counts[init_id] > 1:
            final_id = f"{init_id}_{i}"
            unique_spectra.append(replace(spec, spectrum_id=final_id))
        else:
            unique_spectra.append(spec)

    return unique_spectra


def load_mgf(  # noqa: C901
    file_path: str | Path,
    cache_dir: Path,
    apply_filters: bool = True,
    min_peaks: int = 1,
    use_cache: bool = True,
    num_workers: int | None = None,
) -> list[Spectrum]:
    """Load and parse MGF file into kakapo Spectrum objects in parallel.

    Args:
        file_path: Path to the .mgf file.
        cache_dir: Path to the cache directory.
        apply_filters: If True, apply quality filters to spectra. Default is True.
        min_peaks: Minimum number of peaks a spectrum must have to be included. Default is 1.
        use_cache: If True, check and use the JSON cache for parsed spectra. Default is True.
        num_workers: Number of worker processes for parallel parsing.

    Returns:
        List of kakapo Spectrum objects parsed from the MGF file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid or contains no valid spectra.
    """
    path = Path(file_path)
    logger.info("Loading MGF file from %s", path)

    if not path.exists():
        logger.error("MGF file not found: %s", path)
        raise FileNotFoundError(f"MGF file not found: {path}")

    if path.suffix.lower() != ".mgf":
        logger.error("Expected .mgf file, got: %s", path.suffix)
        raise ValueError(f"Expected .mgf file, got: {path.suffix}")

    source_file_str = str(path.resolve())

    if use_cache:
        try:
            cache_path = _get_cache_path(path, apply_filters, min_peaks, cache_dir)
            if cache_path.exists():
                try:
                    logger.info("Cache hit. Loading spectra from cache %s", cache_path)
                    return _load_from_cache(cache_path)
                except Exception as e:
                    logger.warning("Cache is invalid or corrupted. Re-parsing MGF. Error: %s", e)
        except Exception as e:
            logger.warning("Failed to check cache. Re-parsing MGF. Error: %s", e)

    kakapo_spectra = parse_mgf_file(
        path,
        source_file_path=source_file_str,
        apply_filters=apply_filters,
        min_peaks=min_peaks,
        num_workers=num_workers,
    )

    if not kakapo_spectra:
        logger.warning("No spectra found in %s", path)
        raise ValueError(f"No spectra found in {path}")

    kakapo_spectra = _ensure_unique_spectrum_ids(kakapo_spectra)

    if not kakapo_spectra:
        logger.warning(
            "No valid spectra after filtering in %s (min_peaks threshold: %d)",
            path,
            min_peaks,
        )
        raise ValueError(
            f"No valid spectra after filtering in {path} (min_peaks threshold: {min_peaks})."
        )

    logger.info(
        "Loaded and filtered %d valid spectra from %s.",
        len(kakapo_spectra),
        path,
    )

    if use_cache:
        try:
            cache_path = _get_cache_path(path, apply_filters, min_peaks, cache_dir)
            _save_to_cache(kakapo_spectra, cache_path)
        except Exception as e:
            logger.warning("Failed to save spectra to cache: %s", e)

    return kakapo_spectra


def load_mgf_batch(
    file_paths: list[str | Path],
    cache_dir: Path,
    apply_filters: bool = True,
    min_peaks: int = 1,
    use_cache: bool = True,
    num_workers: int | None = None,
) -> list[Spectrum]:
    """Load and parse multiple MGF files into a single list of Spectrum objects.

    Processes each file sequentially and combines results.

    Args:
        file_paths: List of paths to .mgf files.
        cache_dir: Path to the cache directory.
        apply_filters: If True, apply matchms filtering to each file.
        min_peaks: Minimum number of peaks per spectrum.
        use_cache: If True, check and use the JSON cache for parsed spectra.
        num_workers: number of workers for parallel execution.

    Returns:
        Combined list of kakapo Spectrum objects from all files.
    """
    all_spectra = []
    logger.info("Loading batch of %d MGF file(s)...", len(file_paths))

    for file_path in file_paths:
        try:
            spectra = load_mgf(
                file_path=file_path,
                cache_dir=cache_dir,
                apply_filters=apply_filters,
                min_peaks=min_peaks,
                use_cache=use_cache,
                num_workers=num_workers,
            )
            all_spectra.extend(spectra)
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            logger.warning("Failed to load MGF file %s: %s", file_path, e)
            continue

    logger.info("Completed batch load. Loaded %d total spectra.", len(all_spectra))
    return all_spectra
