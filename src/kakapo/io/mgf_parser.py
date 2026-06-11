"""MGF (Mascot Generic Format) parser for mass spectrometry data.

This module handles:
- Loading .mgf files using matchms
- Filtering and cleaning spectra
- Converting matchms spectra to kakapo Spectrum models
- Extracting precursor mass from both 'precursor_mz' and 'pepmass' fields
- Robust error handling for malformed data
"""

from pathlib import Path

from matchms import Spectrum as MatchmsSpectrum
from matchms.filtering.default_pipelines import CLEAN_PEAKS, DEFAULT_FILTERS
from matchms.importing import load_from_mgf

from kakapo.models import Peak, Spectrum
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def load_mgf(  # noqa: C901
    file_path: str | Path,
    apply_filters: bool = True,
    min_peaks: int = 1,
) -> list[Spectrum]:
    """Load and parse MGF file into kakapo Spectrum objects.

    Uses matchms to load the file and optionally applies standard filtering
    and cleaning operations. Converts matched peaks to kakapo Peak models.

    Args:
        file_path: Path to the .mgf file.
        apply_filters: If True, apply matchms CLEAN_PEAKS and DEFAULT_FILTERS.
                      Default is True.
        min_peaks: Minimum number of peaks a spectrum must have to be included.
                  Default is 1.

    Returns:
        List of kakapo Spectrum objects parsed from the MGF file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid or contains no valid spectra.
        RuntimeError: If spectrum conversion fails.

    Examples:
        >>> spectra = load_mgf("data/spectra.mgf")
        >>> print(f"Loaded {len(spectra)} spectra")

        >>> spectra = load_mgf("data/spectra.mgf", apply_filters=False)
    """
    file_path = Path(file_path)
    logger.info("Loading MGF file from %s", file_path)

    if not file_path.exists():
        logger.error("MGF file not found: %s", file_path)
        raise FileNotFoundError(f"MGF file not found: {file_path}")

    if not file_path.suffix.lower() == ".mgf":
        logger.error("Expected .mgf file, got: %s", file_path.suffix)
        raise ValueError(f"Expected .mgf file, got: {file_path.suffix}")

    try:
        matchms_spectra = list(load_from_mgf(str(file_path)))
    except Exception as e:
        logger.error("Failed to parse MGF file %s: %s", file_path, e)
        raise ValueError(f"Failed to parse MGF file {file_path}: {e}") from e

    if not matchms_spectra:
        logger.warning("No spectra found in %s", file_path)
        raise ValueError(f"No spectra found in {file_path}")

    # Apply filters if requested
    if apply_filters:
        logger.debug("Applying matchms CLEAN_PEAKS and DEFAULT_FILTERS to %s", file_path)
        from matchms import SpectrumProcessor

        processor = SpectrumProcessor(CLEAN_PEAKS + DEFAULT_FILTERS)
        processed_result = processor.process_spectra(matchms_spectra)

        # SpectrumProcessor returns a tuple (spectra, None)
        if isinstance(processed_result, tuple):
            matchms_spectra = processed_result[0]
        else:
            matchms_spectra = processed_result

    # Convert matchms spectra to kakapo models
    kakapo_spectra = []
    for matchms_spec in matchms_spectra:
        spec_id = _get_spectrum_id(matchms_spec.metadata)
        try:
            if len(matchms_spec.peaks) < min_peaks:
                logger.debug(
                    "Skipping spectrum %s (peaks: %d < min_peaks: %d)",
                    spec_id,
                    len(matchms_spec.peaks),
                    min_peaks,
                )
                continue

            spectrum = _matchms_to_kakapo(matchms_spec)
            kakapo_spectra.append(spectrum)
        except (ValueError, KeyError, RuntimeError) as e:
            logger.error("Failed to convert spectrum %s: %s", spec_id, e)
            raise RuntimeError(
                f"Failed to convert spectrum {spec_id}:"
                f" {e}"
            ) from e

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
    return kakapo_spectra


def load_mgf_batch(
    file_paths: list[str | Path],
    apply_filters: bool = True,
    min_peaks: int = 1,
) -> list[Spectrum]:
    """Load and parse multiple MGF files into a single list of Spectrum objects.

    Processes each file sequentially and combines results.

    Args:
        file_paths: List of paths to .mgf files.
        apply_filters: If True, apply matchms filtering to each file.
        min_peaks: Minimum number of peaks per spectrum.

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
            spectra = load_mgf(file_path, apply_filters, min_peaks)
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


def _get_spectrum_id(metadata: dict) -> str:
    """Extract spectrum identifier from metadata.

    Tries multiple common field names and falls back to a generic ID.

    Args:
        metadata: Spectrum metadata dictionary.

    Returns:
        Spectrum identifier string.
    """
    for field in ["spectrum_id", "scan", "spectrumID", "title"]:
        if field in metadata and metadata[field] is not None:
            return str(metadata[field])

    return "unknown"


def _matchms_to_kakapo(matchms_spec: MatchmsSpectrum) -> Spectrum:  # noqa: C901
    """Convert a matchms Spectrum to a kakapo Spectrum.

    Extracts peak data and metadata from matchms format and constructs
    the corresponding kakapo model.

    Args:
        matchms_spec: A spectrum from matchms.

    Returns:
        A kakapo Spectrum instance.

    Raises:
        ValueError: If required data is missing or invalid.
    """
    metadata = matchms_spec.metadata

    # Extract spectrum ID
    spectrum_id = _get_spectrum_id(metadata)
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
    )
