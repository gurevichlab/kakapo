"""MGF (Mascot Generic Format) parser for mass spectrometry data.

This module handles:
- Loading .mgf files using matchms
- Filtering and cleaning spectra
- Converting matchms spectra to kakapo Spectrum models
- Extracting precursor mass from both 'precursor_mz' and 'pepmass' fields
- Robust error handling for malformed data
"""

import hashlib
import json
import logging
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from matchms import Spectrum as MatchmsSpectrum
from matchms import SpectrumProcessor
from matchms.importing import load_from_mgf

from kakapo.config import DEFAULT_FILTERS
from kakapo.models import Peak, Spectrum
from kakapo.utils.logger import get_logger

# Suppress verbose warnings from matchms
logging.getLogger("matchms").setLevel(logging.ERROR)

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
        peaks = frozenset(Peak(mass=p[0], intensity=p[1]) for p in item["peaks"])
        spectra.append(
            Spectrum(
                spectrum_id=item["spectrum_id"],
                source_file_path=item["source_file_path"],
                peaks=peaks,
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
            "peaks": [[p.mass, p.intensity] for p in spec.get_sorted_peaks()],
            "source_file_path": spec.source_file_path,
        }
        for spec in spectra
    ]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _process_and_convert_chunk(
    spectra_chunk: list[MatchmsSpectrum],
    apply_filters: bool,
    min_peaks: int,
    source_file_path: str,
) -> list[Spectrum]:
    """Process a chunk of matchms spectra and convert them to kakapo Spectrum models.

    Args:
        spectra_chunk: A list of matchms Spectrum objects.
        apply_filters: Whether to apply matchms CLEAN_PEAKS.
        min_peaks: Minimum number of peaks required for a spectrum to be included.
        source_file_path: Path to the original MGF file.

    Returns:
        A list of kakapo Spectrum models.
    """
    if apply_filters:
        processor = SpectrumProcessor(DEFAULT_FILTERS)
        processed_result = processor.process_spectra(spectra_chunk, progress_bar=False)
        if isinstance(processed_result, tuple):
            spectra_chunk = processed_result[0]
        else:
            spectra_chunk = processed_result

    kakapo_spectra = []
    for matchms_spec in spectra_chunk:
        spec_id = matchms_spec.metadata.get("spectrum_id")
        try:
            if len(matchms_spec.peaks) < min_peaks:
                logger.debug(
                    "Skipping spectrum %s (peaks: %d < min_peaks: %d)",
                    spec_id,
                    len(matchms_spec.peaks),
                    min_peaks,
                )
                continue

            spectrum = _matchms_to_kakapo(matchms_spec, source_file_path=source_file_path)
            kakapo_spectra.append(spectrum)
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("Failed to convert spectrum %s: %s", spec_id, e)
            raise RuntimeError(f"Failed to convert spectrum {spec_id}: {e}") from e

    return kakapo_spectra


def load_mgf(  # noqa: C901
    file_path: str | Path,
    cache_dir: Path,
    apply_filters: bool = True,
    min_peaks: int = 1,
    use_cache: bool = True,
    num_workers: int | None = None,
) -> list[Spectrum]:
    """Load and parse MGF file into kakapo Spectrum objects.

    Uses matchms to load the file and optionally applies standard filtering
    and cleaning operations. Converts matched peaks to kakapo Peak models.

    Args:
        file_path: Path to the .mgf file.
        cache_dir: Path to the cache directory.
        apply_filters: If True, apply matchms CLEAN_PEAKS.
                      Default is True.
        min_peaks: Minimum number of peaks a spectrum must have to be included.
                  Default is 1.
        use_cache: If True, check and use the JSON cache for parsed spectra.
                   Default is True.

    Returns:
        List of kakapo Spectrum objects parsed from the MGF file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid or contains no valid spectra.
        RuntimeError: If spectrum conversion fails.

    Examples:
        >>> spectra = load_mgf("data/spectra.mgf", Path(".cache"))
        >>> print(f"Loaded {len(spectra)} spectra")
    """
    file_path = Path(file_path)
    logger.info("Loading MGF file from %s", file_path)

    if not file_path.exists():
        logger.error("MGF file not found: %s", file_path)
        raise FileNotFoundError(f"MGF file not found: {file_path}")

    if not file_path.suffix.lower() == ".mgf":
        logger.error("Expected .mgf file, got: %s", file_path.suffix)
        raise ValueError(f"Expected .mgf file, got: {file_path.suffix}")

    source_file_str = str(file_path.resolve())

    if use_cache:
        try:
            cache_path = _get_cache_path(file_path, apply_filters, min_peaks, cache_dir)
            if cache_path.exists():
                try:
                    logger.info("Cache hit. Loading spectra from cache %s", cache_path)
                    return _load_from_cache(cache_path)
                except Exception as e:
                    logger.warning("Cache is invalid or corrupted. Re-parsing MGF. Error: %s", e)
        except Exception as e:
            logger.warning("Failed to check cache. Re-parsing MGF. Error: %s", e)

    try:
        matchms_spectra = list(load_from_mgf(str(file_path)))
    except Exception as e:
        logger.error("Failed to parse MGF file %s: %s", file_path, e)
        raise ValueError(f"Failed to parse MGF file {file_path}: {e}") from e

    if not matchms_spectra:
        logger.warning("No spectra found in %s", file_path)
        raise ValueError(f"No spectra found in {file_path}")

    _ensure_unique_spectrum_ids(matchms_spectra)

    # Convert and process matchms spectra to kakapo models
    num_spectra = len(matchms_spectra)
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    if num_spectra > 100 and num_workers > 1:
        chunk_size = max(100, num_spectra // (num_workers * 4))
        chunks = [matchms_spectra[i : i + chunk_size] for i in range(0, num_spectra, chunk_size)]

        logger.info(
            "Processing %d spectra in parallel using %d workers (chunk size: %d)...",
            num_spectra,
            num_workers,
            chunk_size,
        )

        kakapo_spectra = []
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        _process_and_convert_chunk,
                        chunk,
                        apply_filters,
                        min_peaks,
                        source_file_str,
                    )
                    for chunk in chunks
                ]
                for future in futures:
                    kakapo_spectra.extend(future.result())
        except Exception as e:
            logger.error("Failed parallel processing of MGF spectra: %s", e)
            raise
    else:
        kakapo_spectra = _process_and_convert_chunk(
            matchms_spectra,
            apply_filters,
            min_peaks,
            source_file_str,
        )

    if not kakapo_spectra:
        logger.warning(
            "No valid spectra after filtering in %s (min_peaks threshold: %d)",
            file_path,
            min_peaks,
        )
        raise ValueError(
            f"No valid spectra after filtering in {file_path}. "
            f"(Loaded {len(matchms_spectra)} total, "
            f"min_peaks threshold: {min_peaks})"
        )

    logger.info(
        "Loaded %d spectra from %s. Found %d valid spectra after filtering.",
        len(matchms_spectra),
        file_path,
        len(kakapo_spectra),
    )

    if use_cache:
        try:
            cache_path = _get_cache_path(file_path, apply_filters, min_peaks, cache_dir)
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

    Returns:
        Combined list of kakapo Spectrum objects from all files.

    Raises:
        FileNotFoundError: If any file does not exist.
        ValueError: If any file format is invalid.
        RuntimeError: If any spectrum conversion fails.

    Examples:
        >>> all_spectra = load_mgf_batch(["file1.mgf", "file2.mgf"])
        >>> print(f"Loaded {len(all_spectra)} total spectra")
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


def _get_precursor_mass(metadata: dict) -> float:
    """Extract precursor mass from matchms spectrum metadata.

    Handles multiple possible field names: 'precursor_mz', 'pepmass', etc.
    Attempts to extract the parent ion mass, which may be stored as:
    - A simple float in 'precursor_mz'
    - A float in 'pepmass'
    - A tuple/list with mass as first element in 'pepmass'

    Args:
        metadata: Spectrum metadata dictionary from matchms.

    Returns:
        The precursor mass as a float.

    Raises:
        ValueError: If precursor mass cannot be extracted or is invalid.
    """
    # Try 'precursor_mz' first
    if "precursor_mz" in metadata and metadata["precursor_mz"] is not None:
        try:
            mass = float(metadata["precursor_mz"])
            if mass >= 0:
                return mass
        except (TypeError, ValueError):
            pass

    # Try 'pepmass' next
    if "pepmass" in metadata and metadata["pepmass"] is not None:
        pepmass = metadata["pepmass"]

        # Handle tuple/list format [mass, intensity]
        if isinstance(pepmass, (tuple, list)) and len(pepmass) > 0:
            try:
                mass = float(pepmass[0])
                if mass >= 0:
                    return mass
            except (TypeError, ValueError):
                pass
        else:
            try:
                mass = float(pepmass)  # type: ignore
                if mass >= 0:
                    return mass
            except (TypeError, ValueError):
                pass

    raise ValueError(
        f"Could not extract precursor mass from metadata. Available fields: {list(metadata.keys())}"
    )


def _ensure_unique_spectrum_ids(matchms_spectra: list[MatchmsSpectrum]) -> None:
    """Ensure all spectra have unique IDs.

    If no ID fields are present, generates ID as index.
    If spectrum IDs are not unique, appends suffix _idx (index of spectrum in the file).
    Modifies the matchms spectra metadata in place.

    Args:
        matchms_spectra: List of matchms Spectrum objects.
    """
    initial_ids = []
    for i, spec in enumerate(matchms_spectra):
        metadata = spec.metadata
        spec_id = None
        for field in ["spectrum_id", "scan", "spectrumID", "title", "feature_id"]:
            if field in metadata and metadata[field] is not None:
                val = str(metadata[field]).strip()
                if val:
                    spec_id = val
                    break

        if spec_id is not None:
            initial_ids.append(spec_id)
        else:
            initial_ids.append(str(i))

    id_counts = Counter(initial_ids)

    for i, spec in enumerate(matchms_spectra):
        init_id = initial_ids[i]
        if id_counts[init_id] > 1:
            final_id = f"{init_id}_{i}"
        else:
            final_id = init_id

        spec.set("spectrum_id", final_id)


def _matchms_to_kakapo(  # noqa: C901
    matchms_spec: MatchmsSpectrum, source_file_path: str
) -> Spectrum:
    """Convert a matchms Spectrum to a kakapo Spectrum.

    Extracts peak data and metadata from matchms format and constructs
    the corresponding kakapo model.

    Args:
        matchms_spec: A spectrum from matchms.
        source_file_path: Path to the original MGF file.

    Returns:
        A kakapo Spectrum instance.

    Raises:
        ValueError: If required data is missing or invalid.
    """
    metadata = matchms_spec.metadata

    # Extract spectrum ID
    spectrum_id = metadata.get("spectrum_id")
    if not spectrum_id:
        raise ValueError("Spectrum ID cannot be empty")

    # Extract precursor mass
    precursor_mass = _get_precursor_mass(metadata)

    # Extract peaks
    peaks_array = matchms_spec.peaks
    if peaks_array is None or len(peaks_array) == 0:
        raise ValueError(f"Spectrum {spectrum_id} has no peaks")

    logger.debug(
        "Converting matchms spectrum %s (%d raw peaks, precursor: %f)",
        spectrum_id,
        len(peaks_array),
        precursor_mass,
    )

    # Convert peaks to kakapo Peak objects
    # peaks_array is expected to be a 2D array [m/z, intensity]
    try:
        peak_objects = set()
        for peak in peaks_array:
            if len(peak) >= 2:
                mass, intensity = float(peak[0]), float(peak[1])

                if mass < 0:
                    raise ValueError(f"Invalid peak mass {mass} in spectrum {spectrum_id}")
                if intensity < 0:
                    raise ValueError(
                        f"Invalid peak intensity {intensity} in spectrum {spectrum_id}"
                    )

                peak_objects.add(Peak(mass=mass, intensity=intensity))

        if not peak_objects:
            raise ValueError(f"No valid peaks in spectrum {spectrum_id}")

    except (TypeError, IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse peaks in spectrum {spectrum_id}: {e}") from e

    # Extract charge
    charge = matchms_spec.get("charge")
    if charge is not None:
        try:
            charge = int(charge)
        except (ValueError, TypeError):
            charge = None

    return Spectrum(
        spectrum_id=spectrum_id,
        peaks=frozenset(peak_objects),
        precursor_mass=precursor_mass,
        charge=charge,
        source_file_path=source_file_path,
    )
