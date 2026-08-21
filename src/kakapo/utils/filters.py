"""Custom matchms peak-processing filters for kakapo."""

from typing import Optional

import numpy as np
from matchms import Fragments, Spectrum

from kakapo.constants import MAX_CHARGE, PROTON_MASS

DEFAULT_PRECURSOR_OFFSET = -1.6


def remove_peaks_relative_to_precursor_mz_charge_aware(
    spectrum_in: Optional[Spectrum],
    offset_to_precursor: float = DEFAULT_PRECURSOR_OFFSET,
    clone: Optional[bool] = True,
) -> Optional[Spectrum]:
    """Remove peaks that are higher than the maximum expected fragment m/z based on precursor
    charge.

    For doubly or multiply charged precursor ions, the neutral parent mass is higher
    than the precursor m/z. Singly-charged fragments can exist up to the neutral parent mass.
    This filter calculates the maximum expected fragment m/z using the precursor charge:
    max_mz = precursor_mz * charge - (charge - 1) * PROTON_MASS
    and removes any peaks with m/z values greater than max_mz + offset_to_precursor.

    Args:
        spectrum_in: The input matchms Spectrum object.
        offset_to_precursor: The offset (in Da) added to the calculated maximum fragment m/z.
        clone: Whether to clone the input spectrum.

    Returns:
        The filtered matchms Spectrum object, or None if the input is None.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    precursor_mz = spectrum.get("precursor_mz", None)
    if precursor_mz is None:
        raise ValueError("Undefined 'precursor_mz'.")
    if not isinstance(precursor_mz, (float, int)):
        raise ValueError("Expected 'precursor_mz' to be a scalar number.")

    charge = spectrum.get("charge", None)
    # Default to charge = 1 if unknown or not specified
    effective_charge = int(abs(charge)) if charge and charge > 0 else MAX_CHARGE

    # Calculate maximum possible singly-charged fragment m/z
    max_mz = precursor_mz * effective_charge - (effective_charge - 1) * PROTON_MASS
    cutoff_mz = max_mz + offset_to_precursor

    mzs, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
    peaks_to_keep = mzs <= cutoff_mz
    spectrum.peaks = Fragments(mz=mzs[peaks_to_keep], intensities=intensities[peaks_to_keep])

    return spectrum


def keep_highest_intensity_in_radius(
    spectrum_in: Optional[Spectrum],
    radius: float = 0.02,
    clone: Optional[bool] = True,
) -> Optional[Spectrum]:
    """Keep a peak only if it has the highest intensity in its radius window.

    For each peak at m/z_i, the radius window is defined as [m/z_i - radius, m/z_i + radius].
    A peak is kept if its intensity is greater than or equal to the intensity of all other
    peaks in this window.

    Args:
        spectrum_in: The input matchms Spectrum object.
        radius: The window radius (in Da) around each peak. Default is 0.02 Da.
        clone: Whether to clone the input spectrum.

    Returns:
        The filtered matchms Spectrum object, or None if the input is None.

    Raises:
        ValueError: If radius is less than or equal to zero.
    """
    if spectrum_in is None:
        return None

    if radius <= 0:
        raise ValueError("Radius must be a positive number.")

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if len(spectrum.peaks) == 0:
        return spectrum

    mzs, intensities = spectrum.peaks.mz, spectrum.peaks.intensities

    # Find the range of peak indices within the window [m/z - radius, m/z + radius]
    left_indices = np.searchsorted(mzs, mzs - radius, side="left")
    right_indices = np.searchsorted(mzs, mzs + radius, side="right")

    peaks_to_keep = np.zeros(len(mzs), dtype=bool)
    for i in range(len(mzs)):
        window_intensities = intensities[left_indices[i] : right_indices[i]]
        if intensities[i] == np.max(window_intensities):
            peaks_to_keep[i] = True

    spectrum.peaks = Fragments(mz=mzs[peaks_to_keep], intensities=intensities[peaks_to_keep])

    return spectrum
