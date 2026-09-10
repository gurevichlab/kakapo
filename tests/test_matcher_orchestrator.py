from dataclasses import replace

import numpy as np
import pytest

from kakapo.constants import PROTON_MASS
from kakapo.core.matcher.indexing import build_experimental_spectrum_index
from kakapo.core.matcher.orchestrator import match_spectra, match_spectrum
from kakapo.models import (
    Chirality,
    ModificationMode,
    Monomer,
    RuleShift,
    ScoringMode,
    Spectrum,
    TheoreticalIon,
    TheoreticalSpectrum,
    Tolerance,
)


def make_spectrum(
    peaks_data: list[tuple[float, float]],
    precursor_mass: float = 500.0,
    charge: int | None = 1,
    spectrum_id: str = "spec_1",
    source_file_path: str = "test.mgf",
) -> Spectrum:
    """Helper to create a Spectrum from list of mass-intensity tuples."""
    mzs = np.array([p[0] for p in peaks_data], dtype=np.float64)
    intensities = np.array([p[1] for p in peaks_data], dtype=np.float64)
    charge_val = charge if charge is not None else 1
    precursor_mz = (precursor_mass + PROTON_MASS * charge_val) / charge_val
    return Spectrum(
        spectrum_id=spectrum_id,
        mzs=mzs,
        intensities=intensities,
        precursor_mass=precursor_mz,
        charge=charge,
        source_file_path=source_file_path,
    )


def make_theoretical_spectrum(
    ion_masses: list[float],
    structure_mass: float = 500.0,
    monomer_count: int = 4,
    structure_id: str = "struct_1",
    monomer_mass: float = 125.0,
) -> TheoreticalSpectrum:
    """Helper to create a TheoreticalSpectrum from list of masses."""
    ions = frozenset(
        TheoreticalIon(
            fragment_id=f"frag_{idx}",
            mask=(1 << idx),
            rules_shifts=(RuleShift("TEST", mass - monomer_mass),),
        )
        for idx, mass in enumerate(ion_masses)
    )
    monomers = {
        idx: Monomer(
            norine_name=f"Mon_{idx}",
            residue="X",
            is_methylated=False,
            chirality=Chirality.L,
            mass=monomer_mass,
        )
        for idx in range(monomer_count)
    }
    return TheoreticalSpectrum(
        structure_id=structure_id,
        ions=ions,
        structure_mass=structure_mass,
        monomer_count=monomer_count,
        monomer_mass_vector=(monomer_mass,) * monomer_count,
        monomers=monomers,
    )


class TestSpectrumMatcherOrchestrator:
    """Unit tests for the top-level orchestrator and batch matching execution."""

    def test_match_spectra_pairwise(self):
        """Test matching lists of experimental and theoretical spectra pairwise."""
        spec1 = make_spectrum([(100.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_1")
        spec2 = make_spectrum([(200.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_2")

        theo1 = make_theoretical_spectrum(
            [100.0 - 1.007276], structure_mass=500.0, monomer_count=2, structure_id="struct_1"
        )
        theo2 = make_theoretical_spectrum(
            [200.0 - 1.007276], structure_mass=500.0, monomer_count=2, structure_id="struct_2"
        )
        results = match_spectra(
            [spec1, spec2],
            [theo1, theo2],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            min_shared_peaks=1,
        )

        assert len(results) == 2

        # Key results by (spectrum_id, structure_id)
        results_map = {(r.spectrum_id, r.structure_id): r for r in results}
        assert results_map[("spec_1", "struct_1")].matched_peaks_count == 1
        assert results_map[("spec_2", "struct_2")].matched_peaks_count == 1

    def test_match_spectra_unknown_charge(self):
        """Test matching spectrum with unknown charge (charge=None)."""
        spec = make_spectrum(
            [(500.0, 100.0)], precursor_mass=1000.0, charge=None, spectrum_id="spec_unknown_charge"
        )
        true_precursor_mz = (1000.0 + 2 * PROTON_MASS) / 2

        spec = replace(spec, precursor_mass=true_precursor_mz)

        theo = make_theoretical_spectrum(
            [500.0 - PROTON_MASS], structure_mass=1000.0, monomer_count=2, structure_id="struct_1"
        )

        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            min_shared_peaks=1,
        )

        assert len(results) == 1
        assert results[0].spectrum_id == "spec_unknown_charge"
        assert results[0].spectrum_mass == true_precursor_mz
        assert results[0].matched_peaks_count == 1
        assert results[0].matched_peaks[0].charge == 1

    def test_match_spectra_empty_inputs(self):
        """Test match_spectra with empty input iterables."""
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum([100.0 - 1.007276], structure_mass=500.0, monomer_count=1)

        assert (
            len(match_spectra([], [theo], tolerance=Tolerance(0.01), scoring=ScoringMode.EXP_PEAKS))
            == 0
        )
        assert (
            len(match_spectra([spec], [], tolerance=Tolerance(0.01), scoring=ScoringMode.EXP_PEAKS))
            == 0
        )
        assert (
            len(match_spectra([], [], tolerance=Tolerance(0.01), scoring=ScoringMode.EXP_PEAKS))
            == 0
        )

    @pytest.mark.parametrize(
        "structure_mass,expected_match",
        [
            # Upper bound checks (precursor = 500.0, diff <= 250.0 -> max structure_mass = 750.0)
            (749.99, True),
            (750.0, True),
            (750.01, False),
            # Lower bound checks (precursor = 500.0, diff <= 250.0 -> min structure_mass = 250.0)
            (250.01, True),
            (250.0, True),
            (249.99, False),
        ],
    )
    def test_match_spectra_mass_difference_limits(self, structure_mass, expected_match):
        """Test pairwise matching filters pairs outside MAX_TOTAL_MASS_DIFFERENCE constraint."""
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_1")
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276],
            structure_mass=structure_mass,
            structure_id="struct_test",
            monomer_mass=500.0,
        )
        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            min_shared_peaks=1,
        )
        if expected_match:
            assert len(results) == 1
            assert results[0].structure_id == "struct_test"
        else:
            assert len(results) == 0

    def test_match_spectra_zero_peaks_skipped(self):
        """Test that pairwise matching skips pairs with 0 matched peaks."""
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_1")
        # Structure mass = 700.0, meaning mass_mod = -200.0.
        # Since monomer_mass is only 125.0, this modification is invalid (<=0),
        # so match_spectrum will yield 0 matched peaks.
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276],
            structure_mass=700.0,
            structure_id="struct_zero_peaks",
            monomer_mass=125.0,
        )
        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            min_shared_peaks=1,
        )
        assert len(results) == 0

    def test_match_spectrum_no_hypotheses_returns_none(self):
        """Test that match_spectrum returns None when no hypotheses can be generated."""
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=600.0, spectrum_id="spec_1")
        theo = make_theoretical_spectrum(
            [100.0 - PROTON_MASS],
            structure_mass=500.0,
            structure_id="struct_nomod",
            monomer_count=2,
        )
        # Precursor mass diff is +100.0 > tolerance, mode is NO_MOD -> 0 hypotheses
        res = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            mode=ModificationMode.NO_MOD,
        )
        assert res is None

    def test_match_spectra_negative_monomer_modification_skipped(self):
        """Test that pairwise matching filters out structures where max monomer mass plus
        mass difference is negative.
        """
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0, spectrum_id="spec_1")
        # Structure mass is 700.0 -> mass_mod = 500.0 - 700.0 = -200.0.
        # Monomer mass is 150.0 -> max is 150.0. 150.0 + (-200.0) = -50.0 < 0.
        # This structure should be skipped.
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276],
            structure_mass=700.0,
            structure_id="struct_neg_monomer",
            monomer_mass=150.0,
        )
        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            min_shared_peaks=1,
        )
        assert len(results) == 0

    def test_match_spectra_zero_mass_modification_allowed(self):
        """Test that a mass modification resulting in exactly 0 monomer mass is allowed
        and matched.
        """
        # Precursor neutral mass = 300.0 (m/z ~ 301.007)
        # Structure mass = 500.0. Mass mod = -200.0.
        # Monomer 0 mass = 200.0 -> updated to 200.0 - 200.0 = 0.0 (allowed).
        # Monomer 1 mass = 300.0 -> remains 300.0.
        # Theoretical ion: fragment contains Monomer 1 (mass 300.0).
        # Experimental peak: 300.0 + PROTON_MASS = 301.007276.
        spec = make_spectrum([(301.007276, 1.0)], precursor_mass=300.0, spectrum_id="spec_1")

        # Build theoretical spectrum
        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_0",
                    mask=2,  # Monomer 1 only
                    rules_shifts=(RuleShift("TEST", 0.0),),
                )
            ]
        )
        monomers = {
            0: Monomer(
                norine_name="Mon_0",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=200.0,
            ),
            1: Monomer(
                norine_name="Mon_1",
                residue="Y",
                is_methylated=False,
                chirality=Chirality.L,
                mass=300.0,
            ),
        }
        theo = TheoreticalSpectrum(
            structure_id="struct_zero_mass",
            ions=ions,
            structure_mass=500.0,
            monomer_count=2,
            monomer_mass_vector=(200.0, 300.0),
            monomers=monomers,
        )

        results = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            min_shared_peaks=1,
        )
        assert len(results) == 1
        assert results[0].matched_peaks_count == 1
        assert len(results[0].modifications) == 1
        assert results[0].modifications[0].monomer_idx == 0

    def test_match_spectra_min_shared_peaks_filtering(self):
        """Verify that match_spectra filters out results where matched_peaks_count < min_shared_peaks."""
        theo = make_theoretical_spectrum(
            [100.0 - PROTON_MASS, 200.0 - PROTON_MASS],
            structure_id="struct_min_shared",
            structure_mass=300.0,
            monomer_count=2,
        )

        spec = make_spectrum(
            [(100.0, 1.0), (999.0, 1.0)],
            precursor_mass=300.0 + PROTON_MASS,
            charge=1,
            spectrum_id="spec_min_shared",
        )

        # min_shared_peaks=1 -> match returned
        results_1 = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            min_shared_peaks=1,
        )
        assert len(results_1) == 1
        assert results_1[0].matched_peaks_count == 1

        # min_shared_peaks=2 -> match filtered out
        results_2 = match_spectra(
            [spec],
            [theo],
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            min_shared_peaks=2,
        )
        assert len(results_2) == 0

    def test_indexed_vs_direct_matching_equivalence(self) -> None:
        """Verify that matching with a pre-built spec_index produces identical MatchResult."""
        peaks = [(100.01, 10.0), (200.02, 20.0), (300.03, 30.0), (400.04, 40.0)]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - PROTON_MASS, 200.0 - PROTON_MASS], structure_mass=500.0, monomer_count=2
        )

        res_direct = match_spectrum(
            spec, theo, tolerance=Tolerance(0.05), scoring=ScoringMode.INTENSITY
        )
        spec_idx = build_experimental_spectrum_index(spec)
        res_indexed = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.05),
            scoring=ScoringMode.INTENSITY,
            spec_index=spec_idx,
        )

        assert res_direct is not None
        assert res_indexed is not None
        assert res_direct.matched_peaks_count == res_indexed.matched_peaks_count
        assert res_direct.kakapo_score == pytest.approx(res_indexed.kakapo_score)
        assert len(res_direct.matched_peaks) == len(res_indexed.matched_peaks)

    def test_parallel_vs_sequential_matching_equivalence(self, monkeypatch) -> None:
        """Verify that parallel matching produces identical results to sequential matching."""
        import os

        from kakapo.models import ModificationMode

        peaks_1 = [(100.01, 10.0), (200.02, 20.0), (300.03, 30.0)]
        peaks_2 = [(150.01, 15.0), (250.02, 25.0), (350.03, 35.0)]
        spectra = [
            make_spectrum(peaks_1, precursor_mass=500.0, spectrum_id=f"spec_{i}")
            if i % 2 == 0
            else make_spectrum(peaks_2, precursor_mass=600.0, spectrum_id=f"spec_{i}")
            for i in range(10)
        ]

        theo_spectra = [
            make_theoretical_spectrum(
                [100.0 - PROTON_MASS, 200.0 - PROTON_MASS],
                structure_mass=500.0,
                monomer_count=2,
                structure_id=f"struct_{i}",
            )
            for i in range(5)
        ]

        # Force sequential execution path by mocking cpu_count to 1
        monkeypatch.setattr(os, "cpu_count", lambda: 1)
        res_seq = match_spectra(
            spectra=spectra,
            theoretical_spectra=theo_spectra,
            tolerance=Tolerance(0.05),
            scoring=ScoringMode.INTENSITY,
            min_shared_peaks=1,
            mode=ModificationMode.FREE_MOD,
        )

        # Force parallel execution path by mocking cpu_count to 4
        monkeypatch.setattr(os, "cpu_count", lambda: 4)
        res_para = match_spectra(
            spectra=spectra,
            theoretical_spectra=theo_spectra,
            tolerance=Tolerance(0.05),
            scoring=ScoringMode.INTENSITY,
            min_shared_peaks=1,
            mode=ModificationMode.FREE_MOD,
        )

        assert len(res_seq) == len(res_para)
        res_seq_sorted = sorted(res_seq, key=lambda r: (r.spectrum_id, r.structure_id))
        res_para_sorted = sorted(res_para, key=lambda r: (r.spectrum_id, r.structure_id))

        for r_s, r_p in zip(res_seq_sorted, res_para_sorted):
            assert r_s.spectrum_id == r_p.spectrum_id
            assert r_s.structure_id == r_p.structure_id
            assert r_s.matched_peaks_count == r_p.matched_peaks_count
            assert r_s.kakapo_score == pytest.approx(r_p.kakapo_score)
            assert len(r_s.matched_peaks) == len(r_p.matched_peaks)
