"""Custom matchms peak-processing filters for kakapo."""

from typing import Optional

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
