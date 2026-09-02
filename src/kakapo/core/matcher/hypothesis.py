"""Modification hypothesis generators and evaluator for spectrum matching."""

from collections.abc import Iterable, Iterator

from kakapo.constants import MAX_CHARGE, RESIDUE_MODIFICATIONS
from kakapo.core.matcher.indexing import (
    ExperimentalSpectrumIndex,
    _calculate_theory_ions_with_mass,
)
from kakapo.core.matcher.scoring import (
    _get_matched_peaks_count,
    _match_peaks,
    _match_peaks_intensity,
)
from kakapo.models import (
    MatchedPeak,
    MatchResult,
    ModificationHypothesis,
    MonomerModification,
    ScoringMode,
    TheoreticalIonWithMass,
    TheoreticalSpectrum,
    Tolerance,
)


def _generate_unmodified_hypotheses(
    theoretical_spectrum: TheoreticalSpectrum,
) -> Iterator[ModificationHypothesis]:
    """Generate the single hypothesis for unmodified spectrum matching.

    Args:
        theoretical_spectrum: The TheoreticalSpectrum representing the base structure.

    Yields:
        The unmodified ModificationHypothesis (empty modifications).
    """
    yield ModificationHypothesis(
        updated_monomer_masses=list(theoretical_spectrum.monomer_mass_vector),
        modifications=(),
    )


def _generate_single_mod_hypotheses(
    theoretical_spectrum: TheoreticalSpectrum,
    mass_mod: float,
    x_mode_only: bool,
    tolerance: Tolerance,
) -> Iterator[ModificationHypothesis]:
    """Generate candidate hypotheses for single-modification matching.

    For each monomer, creates a hypothesis where that monomer is modified by the given mass change.

    Args:
        theoretical_spectrum: The TheoreticalSpectrum representing the base structure.
        mass_mod: The total mass difference to apply to a monomer.
        x_mode_only: If True, only allow modifications on monomers starting with 'X'.
        tolerance: The tolerance configuration allowed for a match.

    Yields:
        ModificationHypothesis instances for each valid single modification candidate.
    """
    monomer_masses = theoretical_spectrum.monomer_mass_vector
    for i in range(len(monomer_masses)):
        monomer = theoretical_spectrum.monomers.get(i)
        norine_name = monomer.norine_name if monomer is not None else f"Mon_{i}"

        if x_mode_only and not norine_name.startswith("X"):
            continue

        updated_masses = list(monomer_masses)
        updated_masses[i] += mass_mod
        if updated_masses[i] < -tolerance.get_limit(monomer_masses[i]):
            continue

        mod = MonomerModification(
            monomer_idx=i,
            initial_monomer_name=norine_name,
            initial_monomer_mass=monomer_masses[i],
            mass_difference=mass_mod,
            new_monomer_name=None,
            formula_difference=None,
        )
        yield ModificationHypothesis(updated_monomer_masses=updated_masses, modifications=(mod,))


def _generate_two_mod_hypotheses(
    theoretical_spectrum: TheoreticalSpectrum,
    mass_mod: float,
    tolerance: Tolerance,
) -> Iterator[ModificationHypothesis]:
    """Generate candidate hypotheses for two-modification matching (known mod + residual on X).

    Looks for monomers with known modifications (from RESIDUE_MODIFICATIONS) and pairs them with
    residual shifts on X monomers.

    Args:
        theoretical_spectrum: The TheoreticalSpectrum representing the base structure.
        mass_mod: The total precursor mass difference to distribute between two modifications.
        tolerance: The tolerance configuration allowed for a match.

    Yields:
        ModificationHypothesis instances for each valid two-modification candidate.
    """
    monomer_masses = theoretical_spectrum.monomer_mass_vector
    monomers = theoretical_spectrum.monomers

    for j, monomer_j in monomers.items():
        norine_name = monomer_j.norine_name
        known_mods = RESIDUE_MODIFICATIONS.get(norine_name, [])
        if not known_mods:
            continue

        for mod_entry in known_mods:
            known_mod_mass = float(mod_entry["mass_difference"])
            residual_mass = mass_mod - known_mod_mass

            for k, monomer_k in monomers.items():
                if not monomer_k.norine_name.startswith("X"):
                    continue

                updated_masses = list(monomer_masses)
                updated_masses[j] += known_mod_mass
                updated_masses[k] += residual_mass

                tol_j = tolerance.get_limit(monomer_masses[j])
                tol_k = tolerance.get_limit(monomer_masses[k])
                if updated_masses[j] < -tol_j or updated_masses[k] < -tol_k:
                    continue

                new_name_1 = (
                    mod_entry["modified_monomers"][0]
                    if mod_entry.get("modified_monomers")
                    else None
                )
                mod1 = MonomerModification(
                    monomer_idx=j,
                    initial_monomer_name=monomer_j.norine_name,
                    initial_monomer_mass=monomer_masses[j],
                    mass_difference=known_mod_mass,
                    new_monomer_name=new_name_1,
                    formula_difference=mod_entry.get("formula_difference"),
                )
                mod2 = MonomerModification(
                    monomer_idx=k,
                    initial_monomer_name=monomer_k.norine_name,
                    initial_monomer_mass=monomer_masses[k],
                    mass_difference=residual_mass,
                    new_monomer_name=None,
                    formula_difference=None,
                )
                yield ModificationHypothesis(
                    updated_monomer_masses=updated_masses, modifications=(mod1, mod2)
                )


def _evaluate_hypotheses(
    spec_index: ExperimentalSpectrumIndex,
    theoretical_spectrum: TheoreticalSpectrum,
    hypotheses: Iterable[ModificationHypothesis],
    tolerance: Tolerance,
    scoring: ScoringMode,
    spectrum_id: str,
    precursor_mass: float,
    charge: int | None,
) -> MatchResult:
    """Evaluate candidate modification hypotheses and return the best scoring MatchResult.

    Args:
        spec_index: Pre-indexed experimental spectrum.
        theoretical_spectrum: The TheoreticalSpectrum template representing the structure.
        hypotheses: The candidate modification hypotheses to evaluate.
        tolerance: The tolerance configuration allowed for a match.
        scoring: The scoring mode parameter.
        spectrum_id: The ID of the experimental spectrum.
        precursor_mass: The mass of the experimental precursor.
        charge: The charge state of the experimental spectrum.

    Returns:
        A MatchResult instance for the optimal hypothesis.
    """
    max_charge = charge if charge is not None else MAX_CHARGE

    max_score = float("-inf")
    max_matched_count = 0
    best_matched_peaks: list[MatchedPeak] = []
    best_modifications: tuple[MonomerModification, ...] = ()
    best_theory_ions: list[TheoreticalIonWithMass] = []
    found_any = False

    for hypothesis in hypotheses:
        theory_ions_with_mass = _calculate_theory_ions_with_mass(
            theoretical_spectrum, max_charge, hypothesis.modifications
        )

        if scoring == ScoringMode.INTENSITY:
            score, matched_peaks = _match_peaks_intensity(
                spec_index, theory_ions_with_mass, max_charge, tolerance
            )
            matched_peaks_count = len(matched_peaks)
        else:
            matched_peaks_count, matched_peaks = _match_peaks(
                spec_index, theory_ions_with_mass, max_charge, tolerance
            )
            score = float(_get_matched_peaks_count(scoring, matched_peaks, matched_peaks_count))

        if matched_peaks_count > 0 or scoring == ScoringMode.INTENSITY:
            if not found_any or score > max_score:
                found_any = True
                max_score = score
                max_matched_count = matched_peaks_count
                best_matched_peaks = matched_peaks
                best_modifications = hypothesis.modifications
                best_theory_ions = theory_ions_with_mass

    final_score = max_score if found_any else 0.0

    if not found_any:
        best_theory_ions = _calculate_theory_ions_with_mass(theoretical_spectrum, max_charge, ())

    post_modification_ion_masses = [ion.mz for ion in best_theory_ions]

    return MatchResult(
        structure_id=theoretical_spectrum.structure_id,
        spectrum_id=spectrum_id,
        structure_mass=theoretical_spectrum.structure_mass,
        spectrum_mass=precursor_mass,
        tolerance=tolerance,
        matched_peaks_count=max_matched_count,
        theoretical_peaks_count=len(theoretical_spectrum.ions),
        experimental_peaks_count=len(spec_index.mzs),
        monomer_count=theoretical_spectrum.monomer_count,
        unmodified_monomer_mass_vector=theoretical_spectrum.monomer_mass_vector,
        score=final_score,
        matched_peaks=best_matched_peaks,
        modifications=best_modifications,
        post_modification_ion_masses=post_modification_ion_masses,
    )
