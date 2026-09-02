"""Native NumPy peak-processing filters for kakapo."""

from dataclasses import replace
from typing import Optional

import numpy as np

from kakapo.constants import MAX_CHARGE, PROTON_MASS
from kakapo.models import Spectrum

DEFAULT_PRECURSOR_OFFSET = -1.6


def select_by_relative_intensity(
    spectrum: Optional[Spectrum],
    intensity_from: float = 0.001,
) -> Optional[Spectrum]:
    """Filter out peaks with intensity lower than intensity_from fraction of max intensity.

    Args:
        spectrum: Input kakapo Spectrum object.
        intensity_from: Minimum relative intensity (e.g. 0.001 = 0.1% of max intensity).

    Returns:
        Filtered Spectrum object, or None if input is None or no peaks remain.
    """
    if spectrum is None or len(spectrum.mzs) == 0:
        return spectrum

    max_intensity = np.max(spectrum.intensities)
    cutoff = intensity_from * max_intensity
    mask = spectrum.intensities >= cutoff

    if not np.any(mask):
        return None

    return replace(spectrum, mzs=spectrum.mzs[mask], intensities=spectrum.intensities[mask])


def reduce_to_number_of_peaks(
    spectrum: Optional[Spectrum],
    n_max: int = 1000,
) -> Optional[Spectrum]:
    """Reduce spectrum to at most n_max highest-intensity peaks.

    Args:
        spectrum: Input kakapo Spectrum object.
        n_max: Maximum number of peaks to retain.

    Returns:
        Filtered Spectrum object, or None if input is None.
    """
    if spectrum is None or len(spectrum.mzs) <= n_max:
        return spectrum

    # Find indices of top n_max intensities
    top_indices = np.argpartition(spectrum.intensities, -n_max)[-n_max:]
    # Sort retained indices by m/z to preserve ascending m/z order
    top_indices.sort()

    return replace(
        spectrum,
        mzs=spectrum.mzs[top_indices],
        intensities=spectrum.intensities[top_indices],
    )


def require_minimum_number_of_high_peaks(
    spectrum: Optional[Spectrum],
    no_peaks: int = 5,
    intensity_percent: float = 2.0,
) -> Optional[Spectrum]:
    """Require spectrum to have at least no_peaks peaks above intensity_percent of max intensity.

    Args:
        spectrum: Input kakapo Spectrum object.
        no_peaks: Minimum number of high-intensity peaks required.
        intensity_percent: Minimum intensity threshold percentage (0-100).

    Returns:
        The input Spectrum if condition is met, or None if discarded.
    """
    if spectrum is None or len(spectrum.mzs) == 0:
        return None

    max_intensity = np.max(spectrum.intensities)
    cutoff = (intensity_percent / 100.0) * max_intensity
    high_peaks_count = np.sum(spectrum.intensities >= cutoff)

    if high_peaks_count < no_peaks:
        return None

    return spectrum


def remove_peaks_relative_to_precursor_mz_charge_aware(
    spectrum: Optional[Spectrum],
    offset_to_precursor: float = DEFAULT_PRECURSOR_OFFSET,
) -> Optional[Spectrum]:
    """Remove peaks that are higher than the maximum expected fragment m/z based on precursor charge.

    Args:
        spectrum: The input kakapo Spectrum object.
        offset_to_precursor: The offset (in Da) added to calculated maximum fragment m/z.

    Returns:
        The filtered kakapo Spectrum object, or None if input is None or no peaks remain.
    """
    if spectrum is None:
        return None

    precursor_mz = spectrum.precursor_mass
    if precursor_mz < 0:
        raise ValueError("Undefined precursor mass.")

    charge = spectrum.charge
    effective_charge = int(abs(charge)) if charge and charge > 0 else MAX_CHARGE

    max_mz = precursor_mz * effective_charge - (effective_charge - 1) * PROTON_MASS
    cutoff_mz = max_mz + offset_to_precursor

    mask = spectrum.mzs <= cutoff_mz
    if not np.any(mask):
        return None

    return replace(spectrum, mzs=spectrum.mzs[mask], intensities=spectrum.intensities[mask])


def keep_highest_intensity_in_radius(
    spectrum: Optional[Spectrum],
    radius: float = 0.02,
) -> Optional[Spectrum]:
    """Keep a peak only if it has the highest intensity within radius window [m/z - radius, m/z + radius].

    Args:
        spectrum: The input kakapo Spectrum object.
        radius: The window radius (in Da) around each peak.

    Returns:
        The filtered kakapo Spectrum object, or None if input is None or no peaks remain.

    Raises:
        ValueError: If radius is <= 0.
    """
    if spectrum is None:
        return None

    if radius <= 0:
        raise ValueError("Radius must be a positive number.")

    if len(spectrum.mzs) == 0:
        return spectrum

    mzs = spectrum.mzs
    intensities = spectrum.intensities
    n = len(mzs)
    if n <= 1:
        return spectrum

    left_indices = np.searchsorted(mzs, mzs - radius, side="left")
    right_indices = np.searchsorted(mzs, mzs + radius, side="right")

    mask = np.ones(n, dtype=bool)
    for i in range(n):
        left_idx, right_idx = left_indices[i], right_indices[i]
        if right_idx - left_idx > 1:
            if intensities[i] < np.max(intensities[left_idx:right_idx]):
                mask[i] = False

    if not np.any(mask):
        return None

    return replace(spectrum, mzs=mzs[mask], intensities=intensities[mask])
