"""Spectrum scoring engine for matching experimental against theoretical data."""

from kakapo.constants import PROTON_MASS
from kakapo.models import MatchResult, Spectrum, TheoreticalSpectrum


def match_spectrum(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    mass_tolerance: float,
) -> MatchResult:
    """Match an experimental spectrum to a theoretical spectrum.

    Performs a greedy peak matching algorithm where each experimental peak can
    match at most one theoretical ion, and vice versa. It counts the number of
    theoretical ions that have a matching experimental peak within the specified
    mass tolerance.

    Args:
        spectrum: The experimental Spectrum instance.
        theoretical_spectrum: The TheoreticalSpectrum instance.
        mass_tolerance: The maximum absolute mass difference (in Da) allowed for a match.

    Returns:
        A MatchResult instance containing matching metrics.
    """
    # Sort to ensure reproducible greedy matching
    exp_peaks = sorted(spectrum.peaks, key=lambda p: p.mass)
    theory_ions = sorted(theoretical_spectrum.ions, key=lambda ion: ion.mass)

    matched_peaks_count = 0
    last_idx = 0

    for ion in theory_ions:
        mz = ion.mass + PROTON_MASS
        for idx in range(last_idx, len(exp_peaks)):
            peak = exp_peaks[idx]
            if abs(peak.mass - mz) <= mass_tolerance:
                matched_peaks_count += 1
                last_idx = idx + 1
                break  # no diff minimization, just take the first one
            if peak.mass > ion.mass + mass_tolerance:
                last_idx = idx
                break
        else:
            last_idx += 1

    return MatchResult(
        structure_id=theoretical_spectrum.structure_id,
        spectrum_id=spectrum.spectrum_id,
        structure_mass=theoretical_spectrum.structure_mass,
        spectrum_mass=spectrum.precursor_mass,
        mass_tolerance=mass_tolerance,
        matched_peaks_count=matched_peaks_count,
        theoretical_peaks_count=len(theoretical_spectrum.ions),
        monomer_count=theoretical_spectrum.monomer_count,
    )
