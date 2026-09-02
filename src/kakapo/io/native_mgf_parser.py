"""Native MGF (Mascot Generic Format) parser for mass spectrometry data.

Provides fast, robust, parallel parsing of MGF files directly into native Kakapo data structures
without external dependencies.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

from kakapo.config import DEFAULT_FILTERS
from kakapo.models import Spectrum
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def parse_charge(val: Any) -> Optional[int]:  # noqa: C901
    """Parse charge state from various formats into an integer or None.

    Args:
        val: The input charge value (int, float, str, or None).

    Returns:
        Integer charge value or None if unparseable or zero.
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        try:
            int_val = int(val)
            return int_val if int_val != 0 else None
        except (ValueError, OverflowError):
            return None

    if not isinstance(val, str):
        return None

    s = val.strip().replace(" ", "")
    if not s:
        return None

    sign = 1
    if s.endswith("+"):
        s = s[:-1]
    elif s.endswith("-"):
        sign = -1
        s = s[:-1]

    if s.startswith("+"):
        s = s[1:]
    elif s.startswith("-"):
        sign = -1
        s = s[1:]

    try:
        float_val = float(s)
        int_val = int(float_val) * sign
        return int_val if int_val != 0 else None
    except ValueError:
        return None


def parse_precursor_mass(val: Any) -> Optional[float]:
    """Parse precursor m/z from metadata.

    Args:
        val: The precursor mass value or string.

    Returns:
        Float precursor mass, or None if invalid/unparseable.
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        return float(val) if val >= 0 else None

    if not isinstance(val, str):
        return None

    s = val.strip()
    if not s:
        return None

    first_part = s.split()[0]
    try:
        mass = float(first_part)
        return mass if mass >= 0 else None
    except ValueError:
        return None


def parse_spectrum_id(metadata: dict[str, str], fallback_idx: int) -> str:
    """Extract a unique spectrum ID from spectrum metadata.

    Prioritizes TITLE first, then explicit metadata ID keys, then a fallback index ID.

    Args:
        metadata: Dictionary of metadata key-values parsed from MGF header.
        fallback_idx: Fallback 1-based index if no ID can be inferred.

    Returns:
        The extracted spectrum ID string.
    """
    title = metadata.get("TITLE", "").strip()
    if title:
        return title

    for key in ("SPECTRUM_ID", "ID", "FEATURE_ID", "FEATURE", "SCANS", "SCAN"):
        if key in metadata and metadata[key].strip():
            return metadata[key].strip()

    return f"spectrum_{fallback_idx}"


def _parse_mgf_block(
    body_text: str,
    source_str: str,
    spectrum_idx: int,
) -> Optional[Spectrum]:
    """Parse a single MGF spectrum block body into a kakapo Spectrum object."""
    metadata: dict[str, str] = {}
    mz_list: list[float] = []
    int_list: list[float] = []

    for raw_line in body_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";", "//")):
            continue

        if "=" in line:
            eq_pos = line.find("=")
            key = line[:eq_pos].strip().upper()
            val = line[eq_pos + 1 :].strip()
            metadata[key] = val
        else:
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    mz = float(parts[0])
                    intensity = float(parts[1])
                    if mz >= 0 and intensity >= 0:
                        mz_list.append(mz)
                        int_list.append(intensity)
                except ValueError:
                    pass

    precursor_str = (
        metadata.get("PEPMASS")
        or metadata.get("PRECURSOR_MZ")
        or metadata.get("PRECURSORMZ")
        or metadata.get("PMASS")
    )
    precursor_mass = parse_precursor_mass(precursor_str)
    if precursor_mass is None or not mz_list:
        return None

    charge = parse_charge(metadata.get("CHARGE"))
    spec_id = parse_spectrum_id(metadata, spectrum_idx)

    try:
        mzs = np.array(mz_list, dtype=np.float64)
        intensities = np.array(int_list, dtype=np.float64)
        return Spectrum(
            spectrum_id=spec_id,
            source_file_path=source_str,
            mzs=mzs,
            intensities=intensities,
            precursor_mass=precursor_mass,
            charge=charge,
        )
    except Exception as e:
        logger.warning("Failed to parse spectrum block %s: %s", spec_id, e)
        return None


def apply_spectrum_filters(
    spectrum: Optional[Spectrum],
    filters: list = DEFAULT_FILTERS,
) -> Optional[Spectrum]:
    """Apply a pipeline of filter functions to a single spectrum."""
    curr = spectrum
    for item in filters:
        if curr is None:
            break
        if isinstance(item, tuple):
            func, kwargs = item[0], item[1]
            curr = func(curr, **kwargs)
        else:
            curr = item(curr)
    return curr


def _parse_mgf_chunk_worker(
    args: tuple[list[str], str, int, bool, int],
) -> list[Spectrum]:
    """Worker function to parse a list of raw MGF block bodies in parallel.

    Args:
        args: Tuple of (block_bodies, source_str, start_idx, apply_filters, min_peaks).

    Returns:
        List of parsed Spectrum objects.
    """
    block_bodies, source_str, start_idx, apply_filters, min_peaks = args
    spectra: list[Spectrum] = []
    for i, body in enumerate(block_bodies):
        spec = _parse_mgf_block(body, source_str, start_idx + i + 1)
        if spec is not None:
            if apply_filters:
                spec = apply_spectrum_filters(spec, DEFAULT_FILTERS)

            if spec is not None and len(spec.mzs) >= min_peaks:
                spectra.append(spec)
    return spectra


def parse_mgf_file(  # noqa: C901
    file_path: str | Path,
    source_file_path: Optional[str] = None,
    apply_filters: bool = False,
    min_peaks: int = 1,
    num_workers: Optional[int] = None,
) -> list[Spectrum]:
    """Parse an MGF file into a list of kakapo Spectrum models in parallel.

    Args:
        file_path: Path to the MGF file.
        source_file_path: Optional string representation of the source path.
        apply_filters: If True, apply spectrum filters inside worker processes.
        min_peaks: Minimum peaks threshold per spectrum.
        num_workers: Optional number of worker processes for parallel parsing.

    Returns:
        List of parsed kakapo Spectrum objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is not an MGF or contains no valid spectra.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MGF file not found: {path}")

    source_str = source_file_path if source_file_path is not None else str(path.resolve())

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        logger.error("Error reading MGF file %s: %s", path, e)
        raise ValueError(f"Error reading MGF file {path}: {e}") from e

    raw_blocks = content.split("BEGIN IONS")
    if len(raw_blocks) <= 1:
        logger.warning("No spectrum blocks ('BEGIN IONS') found in %s", path)
        return []

    block_bodies = []
    for raw_block in raw_blocks[1:]:
        if "END IONS" in raw_block:
            body = raw_block.split("END IONS")[0]
            block_bodies.append(body)

    total_blocks = len(block_bodies)
    if total_blocks == 0:
        return []

    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)

    if num_workers <= 1 or total_blocks < 50:
        return _parse_mgf_chunk_worker((block_bodies, source_str, 0, apply_filters, min_peaks))

    chunk_size = (total_blocks + num_workers - 1) // num_workers
    work_items = []
    for i in range(0, total_blocks, chunk_size):
        chunk_bodies = block_bodies[i : i + chunk_size]
        work_items.append((chunk_bodies, source_str, i, apply_filters, min_peaks))

    spectra: list[Spectrum] = []
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = executor.map(_parse_mgf_chunk_worker, work_items)
            for res in results:
                spectra.extend(res)
    except Exception as e:
        logger.error("Parallel MGF parsing failed: %s", e)
        raise ValueError(f"Parallel MGF parsing failed: {e}") from e

    return spectra
