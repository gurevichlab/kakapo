"""Spectrum scoring engine for matching experimental against theoretical data."""

from collections.abc import Iterable

from kakapo.config import MAX_TOTAL_MASS_DIFFERENCE
from kakapo.constants import MAX_CHARGE, PROTON_MASS
from kakapo.models import MatchedPeak, MatchResult, Peak, Spectrum, TheoreticalSpectrum
from kakapo.utils.logger import get_logger

logger = get_logger(__name__)


def _fast_dot_product(vector: Iterable[float], mask: int) -> float:
    """Fast dot product of a float vector and a bitmask.

    Args:
        vector: An iterable of floats (e.g. monomer masses).
        mask: An integer bitmask representing which monomers are present.

    Returns:
        The sum of the masses of the active monomers.
    """
    total = 0.0
    for val in vector:
        if not mask:
            break
        if mask & 1:
            total += val
        mask >>= 1
    return total


def _match_peaks(
    exp_peaks: list[Peak],
    theory_ions_with_mass: list[tuple[str, float]],
    max_charge: int,
    mass_tolerance: float,
) -> tuple[int, list[MatchedPeak]]:
    """Greedy peak matching algorithm of pre-calculated theoretical ions against experimental
    peaks.

    Args:
        exp_peaks: Sorted list of experimental peaks.
        theory_ions_with_mass: Sorted list of theoretical ions with pre-calculated masses.
        max_charge: Maximum charge to consider.
        mass_tolerance: Maximum absolute mass difference allowed for a match.

    Returns:
        Tuple of (matched_peaks_count, matched_peaks).
    """
    matched_peaks_count = 0
    matched_peaks: list[MatchedPeak] = []

    for charge in range(1, max_charge + 1):
        # Two pointers technique to find matches
        last_idx = 0
        for frag_id, ion_mass in theory_ions_with_mass:
            mz = (ion_mass + PROTON_MASS * charge) / charge
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
                        frag_id,
                        mz,
                        charge,
                        peak.mass,
                    )
                    break
                if peak.mass > mz + mass_tolerance:
                    last_idx = idx
                    break
            else:
                last_idx += 1
    return matched_peaks_count, matched_peaks


def _calculate_theory_ions_with_mass(
    theoretical_spectrum: TheoreticalSpectrum,
    monomer_masses: Iterable[float],
) -> list[tuple[str, float]]:
    """Calculate and sort theoretical ion masses based on monomer masses."""
    return sorted(
        [
            (
                ion.fragment_id,
                _fast_dot_product(monomer_masses, ion.mask) + ion.total_shift_mass,
            )
            for ion in theoretical_spectrum.ions
        ],
        key=lambda item: item[1],
    )


def _match_spectrum_unmodified(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    mass_tolerance: float,
    mass_mod: float,
) -> MatchResult:
    """Helper to match spectrum using unmodified monomer masses."""
    # Sort to ensure reproducible greedy matching
    exp_peaks = sorted(spectrum.peaks, key=lambda p: p.mass)
    monomer_masses = theoretical_spectrum.monomer_mass_vector
    max_charge = spectrum.charge if spectrum.charge else MAX_CHARGE

    logger.debug(
        "Spectrum '%s' peaks: %d. Theoretical spectrum '%s' ions: %d. Max charge: %d. "
        "Precursor: %f, Structure: %f, mass_mod: %f",
        spectrum.spectrum_id,
        len(exp_peaks),
        theoretical_spectrum.structure_id,
        len(theoretical_spectrum.ions),
        max_charge,
        spectrum.precursor_mass,
        theoretical_spectrum.structure_mass,
        mass_mod,
    )

    theory_ions_with_mass = _calculate_theory_ions_with_mass(theoretical_spectrum, monomer_masses)

    matched_peaks_count, matched_peaks = _match_peaks(
        exp_peaks, theory_ions_with_mass, max_charge, mass_tolerance
    )

    logger.debug(
        "Matching complete for '%s' vs '%s': matched %d peaks (unmodified).",
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
        experimental_peaks_count=len(spectrum.peaks),
        monomer_count=theoretical_spectrum.monomer_count,
        matched_peaks=matched_peaks,
        modified_monomer_idx=None,
        mass_mod=mass_mod,
    )


def _match_spectrum_with_mod(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    mass_tolerance: float,
    mass_mod: float,
) -> MatchResult:
    """Helper to match spectrum using modification-tolerant monomer mass updates."""
    # Sort to ensure reproducible greedy matching
    exp_peaks = sorted(spectrum.peaks, key=lambda p: p.mass)
    monomer_masses = theoretical_spectrum.monomer_mass_vector
    max_charge = spectrum.charge if spectrum.charge else MAX_CHARGE

    logger.debug(
        "Spectrum '%s' peaks: %d. Theoretical spectrum '%s' ions: %d. Max charge: %d. "
        "Precursor: %f, Structure: %f, mass_mod: %f",
        spectrum.spectrum_id,
        len(exp_peaks),
        theoretical_spectrum.structure_id,
        len(theoretical_spectrum.ions),
        max_charge,
        spectrum.precursor_mass,
        theoretical_spectrum.structure_mass,
        mass_mod,
    )

    max_matched_count = 0
    best_matched_peaks = []
    best_i = None

    for i in range(len(monomer_masses)):
        updated_monomer_masses = list(monomer_masses)
        updated_monomer_masses[i] += mass_mod
        if updated_monomer_masses[i] <= 0:
            continue

        theory_ions_with_mass = _calculate_theory_ions_with_mass(
            theoretical_spectrum, updated_monomer_masses
        )

        matched_peaks_count, matched_peaks = _match_peaks(
            exp_peaks, theory_ions_with_mass, max_charge, mass_tolerance
        )

        if matched_peaks_count > max_matched_count:
            max_matched_count = matched_peaks_count
            best_matched_peaks = matched_peaks
            best_i = i

    logger.debug(
        "Matching complete for '%s' vs '%s': max matched %d peaks (best monomer index: %s).",
        spectrum.spectrum_id,
        theoretical_spectrum.structure_id,
        max_matched_count,
        str(best_i),
    )

    return MatchResult(
        structure_id=theoretical_spectrum.structure_id,
        spectrum_id=spectrum.spectrum_id,
        structure_mass=theoretical_spectrum.structure_mass,
        spectrum_mass=spectrum.precursor_mass,
        mass_tolerance=mass_tolerance,
        matched_peaks_count=max_matched_count,
        theoretical_peaks_count=len(theoretical_spectrum.ions),
        experimental_peaks_count=len(spectrum.peaks),
        monomer_count=theoretical_spectrum.monomer_count,
        matched_peaks=best_matched_peaks,
        modified_monomer_idx=best_i,
        mass_mod=mass_mod,
    )


def match_spectrum(
    spectrum: Spectrum,
    theoretical_spectrum: TheoreticalSpectrum,
    mass_tolerance: float,
) -> MatchResult:
    """Match an experimental spectrum to a theoretical spectrum.

    If the precursor mass of the spectrum matches the structure mass within tolerance,
    it runs the unmodified matching algorithm without searching for modifications.
    Otherwise, it runs the modification-tolerant match algorithm, searching for
    the best monomer to modify by mass_mod = precursor - structure mass.

    Args:
        spectrum: The experimental Spectrum instance.
        theoretical_spectrum: The TheoreticalSpectrum instance.
        mass_tolerance: The maximum absolute mass difference (in Da) allowed for a match.

    Returns:
        A MatchResult instance containing matching metrics.
    """
    logger.debug(
        "Matching experimental spectrum '%s' against structure '%s' (tolerance: %f Da)",
        spectrum.spectrum_id,
        theoretical_spectrum.structure_id,
        mass_tolerance,
    )

    mass_mod = spectrum.precursor_mass - theoretical_spectrum.structure_mass - PROTON_MASS

    if abs(mass_mod) <= mass_tolerance:
        logger.debug(
            "Precursor mass (%f) matches structure mass (%f) within tolerance (%f). "
            "Running unmodified matching.",
            spectrum.precursor_mass,
            theoretical_spectrum.structure_mass,
            mass_tolerance,
        )
        return _match_spectrum_unmodified(spectrum, theoretical_spectrum, mass_tolerance, mass_mod)

    return _match_spectrum_with_mod(spectrum, theoretical_spectrum, mass_tolerance, mass_mod)


def match_spectra(
    spectra: Iterable[Spectrum],
    theoretical_spectra: Iterable[TheoreticalSpectrum],
    mass_tolerance: float,
) -> list[MatchResult]:
    """Match each experimental spectrum to each theoretical spectrum pairwise.

    Args:
        spectra: An iterable of experimental Spectrum instances.
        theoretical_spectra: An iterable of TheoreticalSpectrum instances.
        mass_tolerance: The maximum absolute mass difference (in Da) allowed for a match.

    Returns:
        A list of MatchResult instances for all pairwise matches.
    """
    logger.info("Starting batch matching (cross-matching each spectrum against each structure)...")
    spectra_list = list(spectra)
    theo_list = list(theoretical_spectra)
    logger.debug(
        "Loaded %d experimental spectra and %d theoretical spectra for matching",
        len(spectra_list),
        len(theo_list),
    )

    results: list[MatchResult] = []
    for spec in spectra_list:
        for theo in theo_list:
            if abs(theo.structure_mass - spec.precursor_mass) <= MAX_TOTAL_MASS_DIFFERENCE:
                result = match_spectrum(spec, theo, mass_tolerance)
                results.append(result)

    logger.info(
        "Completed batch matching. Generated %d total MatchResult(s).",
        len(results),
    )
    return results
