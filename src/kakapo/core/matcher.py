"""Spectrum scoring engine for matching experimental against theoretical data."""

from kakapo.constants import MAX_CHARGE, PROTON_MASS
from kakapo.models import MatchedPeak, MatchResult, Spectrum, TheoreticalSpectrum
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def match_spectrum(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    mass_tolerance: float,
) -> MatchResult:
    """Match an experimental spectrum to a theoretical spectrum.

    Performs a greedy peak matching algorithm where each experimental peak can match at most one
    theoretical ion for one charge value, and vice versa. It counts the number of theoretical ions
    that have a matching experimental peak within the specified mass tolerance.

    Note: one experimental peak can match two theoretical ions with different charge values and
    vice versa.

    Args:
        spectrum: The experimental Spectrum instance.
        theoretical_spectrum: The TheoreticalSpectrum instance.
        mass_tolerance: The maximum absolute mass difference (in Da) allowed for a match.

    Returns:
        A MatchResult instance containing matching metrics.
    """
    logger.info(
        "Matching experimental spectrum '%s' against structure '%s' (tolerance: %f Da)",
        spectrum.spectrum_id,
        theoretical_spectrum.structure_id,
        mass_tolerance,
    )

    # Sort to ensure reproducible greedy matching
    exp_peaks = sorted(spectrum.peaks, key=lambda p: p.mass)
    theory_ions = sorted(theoretical_spectrum.ions, key=lambda ion: ion.mass)

    matched_peaks_count = 0
    matched_peaks: list[MatchedPeak] = []
    max_charge = spectrum.charge if spectrum.charge else MAX_CHARGE

    logger.debug(
        "Spectrum '%s' peaks count: %d. Theoretical spectrum '%s' ions count: %d. Max charge: %d",
        spectrum.spectrum_id,
        len(exp_peaks),
        theoretical_spectrum.structure_id,
        len(theory_ions),
        max_charge,
    )

    for charge in range(1, max_charge + 1):
        last_idx = 0
        for ion in theory_ions:
            mz = (ion.mass + PROTON_MASS * charge) / charge
            for idx in range(last_idx, len(exp_peaks)):
                peak = exp_peaks[idx]
                if abs(peak.mass - mz) <= mass_tolerance:
                    matched_peaks_count += 1
                    last_idx = idx + 1
                    matched_peaks.append(
                        MatchedPeak(
                            theoretical_mz=mz,
                            experimental_mz=peak.mass,
                            charge=charge,
                        )
                    )
                    logger.debug(
                        "Matched ion '%s' (m/z: %f, charge: %d) to peak (m/z: %f)",
                        ion.fragment_id,
                        mz,
                        charge,
                        peak.mass,
                    )
                    break  # no diff minimization, just take the first one
                if peak.mass > mz + mass_tolerance:
                    last_idx = idx
                    break
            else:
                last_idx += 1

    logger.info(
        "Matching complete for '%s' vs '%s': matched %d peaks.",
        spectrum.spectrum_id,
        theoretical_spectrum.structure_id,
        matched_peaks_count,
    )

    return MatchResult(
        structure_id=theoretical_spectrum.structure_id,
        spectrum_id=spectrum.spectrum_id,
        structure_mass=theoretical_spectrum.structure_mass,
        spectrum_mass=spectrum.precursor_mass,
        mass_tolerance=mass_tolerance,
        matched_peaks_count=matched_peaks_count,
        theoretical_peaks_count=len(theoretical_spectrum.ions),
        monomer_count=theoretical_spectrum.monomer_count,
        matched_peaks=matched_peaks,
    )
