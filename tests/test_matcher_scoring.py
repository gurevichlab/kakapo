import numpy as np
import pytest

from kakapo.constants import PROTON_MASS
from kakapo.core.matcher.orchestrator import match_spectrum
from kakapo.models import (
    Chirality,
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


class TestSpectrumMatcherScoring:
    """Unit tests for the spectrum matching engine scoring functionality."""

    def test_basic_matching_exact(self):
        """Test basic peak matching where peaks match the theoretical ions exactly."""
        peaks = [(100.0, 1.0), (200.0, 1.0), (300.0, 1.0)]
        ion_masses = [100.0 - 1.007276, 200.0 - 1.007276, 300.0 - 1.007276]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.EXP_PEAKS
        )
        assert result is not None
        assert result.matched_peaks_count == 3

    def test_basic_matching_within_tolerance(self):
        """Test basic peak matching where peaks are slightly off but within the tolerance window."""
        peaks = [(100.01, 1.0), (200.02, 1.0)]
        ion_masses = [100.0 - 1.007276, 200.0 - 1.007276]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.03), scoring=ScoringMode.EXP_PEAKS
        )
        assert result is not None
        assert result.matched_peaks_count == 2

    def test_basic_matching_outside_tolerance(self):
        """Test basic peak matching where peaks are outside the tolerance window."""
        peaks = [(100.01, 1.0), (200.02, 1.0)]
        ion_masses = [100.0 - 1.007276, 200.0 - 1.007276]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.005), scoring=ScoringMode.EXP_PEAKS
        )
        assert result is not None
        assert result.matched_peaks_count == 0

    def test_basic_matching_no_matches(self):
        """Test basic peak matching where theoretical ions do not match any peaks."""
        peaks = [(150.0, 1.0), (250.0, 1.0)]
        ion_masses = [100.0 - 1.007276, 200.0 - 1.007276]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(spec, theo, tolerance=Tolerance(0.1), scoring=ScoringMode.EXP_PEAKS)
        assert result is not None
        assert result.matched_peaks_count == 0

    def test_greedy_matching_unique_peaks(self):
        """Ensure each experimental peak can match at most one theoretical ion, and vice versa."""
        # Two theoretical ions close to the same experimental peak
        spec = make_spectrum([(100.0, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [99.99 - 1.007276, 100.01 - 1.007276],
            structure_mass=500.0,
            monomer_count=2,
        )

        # Only 1 match should occur because there is only 1 experimental peak
        result1 = match_spectrum(
            spec, theo, tolerance=Tolerance(0.05), scoring=ScoringMode.EXP_PEAKS
        )
        assert result1 is not None
        assert result1.matched_peaks_count == 1

        # Two experimental peaks close to the same theoretical ion
        spec2 = make_spectrum([(99.99, 1.0), (100.01, 1.0)], precursor_mass=500.0)
        theo2 = make_theoretical_spectrum([100.0 - 1.007276], structure_mass=500.0, monomer_count=1)

        # Only 1 match should occur because there is only 1 theoretical ion
        result2 = match_spectrum(
            spec2, theo2, tolerance=Tolerance(0.05), scoring=ScoringMode.EXP_PEAKS
        )
        assert result2 is not None
        assert result2.matched_peaks_count == 1

    def test_greedy_matching_takes_first(self):
        """Ensure greedy matching takes the first available peaks."""
        spec = make_spectrum([(100.01, 1.0), (100.05, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276, 100.04 - 1.007276],
            structure_mass=500.0,
            monomer_count=2,
        )

        # 100.0 matches 100.01 (first)
        # 100.04 matches 100.05 (remaining)
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.02), scoring=ScoringMode.EXP_PEAKS
        )
        assert result is not None
        assert result.matched_peaks_count == 2

    def test_matched_peaks_populated(self):
        """Verify matched_peaks details are populated in MatchResult."""
        spec = make_spectrum([(100.01, 1.0), (200.0, 1.0), (300.05, 1.0)], precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            [100.0 - 1.007276, 150.0 - 1.007276, 300.04 - 1.007276],
            structure_mass=500.0,
            monomer_count=3,
        )

        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.02), scoring=ScoringMode.EXP_PEAKS
        )
        assert result is not None
        assert result.matched_peaks_count == 2
        assert result.experimental_peaks_count == 3
        assert len(result.matched_peaks) == 2

        # Check first matched peak
        ep1 = result.matched_peaks[0]
        assert ep1.theoretical_mz == pytest.approx(100.0)
        assert ep1.experimental_mz == pytest.approx(100.01)
        assert ep1.experimental_peak_intensity == pytest.approx(1.0)
        assert ep1.charge == 1

        # Check second matched peak
        ep2 = result.matched_peaks[1]
        assert ep2.theoretical_mz == pytest.approx(300.04)
        assert ep2.experimental_mz == pytest.approx(300.05)
        assert ep2.experimental_peak_intensity == pytest.approx(1.0)
        assert ep2.charge == 1

    @pytest.mark.parametrize(
        "spec_charge,expected_count,expected_charge",
        [
            (1, 0, None),
            (2, 1, 2),
            (None, 1, 2),  # Fallback to MAX_CHARGE = 2
        ],
    )
    def test_charge_matching_limits(self, spec_charge, expected_count, expected_charge):
        """Test matching charge limit restrictions and fallback behavior."""
        theo = make_theoretical_spectrum([198.0], structure_mass=500.0, monomer_count=1)
        spec = make_spectrum([(100.007276, 1.0)], precursor_mass=500.0, charge=spec_charge)
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.EXP_PEAKS
        )
        assert result is not None
        assert result.matched_peaks_count == expected_count
        if expected_count > 0:
            assert result.matched_peaks[0].charge == expected_charge

    def test_matching_multiple_charges(self):
        """Test matching peaks of different charges when spectrum charge allows it."""
        theo = make_theoretical_spectrum([198.0, 298.0], structure_mass=500.0, monomer_count=2)
        spec = make_spectrum([(199.007276, 1.0), (150.007276, 1.0)], precursor_mass=500.0, charge=2)
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.EXP_PEAKS
        )

        assert result is not None
        assert result.matched_peaks_count == 2

        # Sort matched peaks by charge for deterministic assertions
        sorted_matched = sorted(result.matched_peaks, key=lambda p: p.charge)

        # Check first matched peak (charge 1)
        assert sorted_matched[0].charge == 1
        assert sorted_matched[0].theoretical_mz == pytest.approx(199.007276)
        assert sorted_matched[0].experimental_mz == pytest.approx(199.007276)

        # Check second matched peak (charge 2)
        assert sorted_matched[1].charge == 2
        assert sorted_matched[1].theoretical_mz == pytest.approx(150.007276)
        assert sorted_matched[1].experimental_mz == pytest.approx(150.007276)

    def test_scoring_modes(self):
        """Test exp-peaks scoring mode ensures each experimental peak is matched at most once."""
        # Create a spectrum with a single peak at 100.0, max charge 2
        spec = make_spectrum(
            [(100.0, 1.0)],
            precursor_mass=250.0,
            charge=2,
            spectrum_id="spec_test",
        )

        # Create theoretical spectrum with two ions that will both match 100.0
        # under charge 1 and charge 2 respectively.
        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_1",
                    mask=1,
                    rules_shifts=(RuleShift("TEST", 100.0 - PROTON_MASS - 125.0),),
                ),
                TheoreticalIon(
                    fragment_id="frag_2",
                    mask=2,
                    rules_shifts=(RuleShift("TEST", 200.0 - 2 * PROTON_MASS - 125.0),),
                ),
            ]
        )
        monomers = {
            0: Monomer(
                norine_name="Mon_0",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=125.0,
            ),
            1: Monomer(
                norine_name="Mon_1",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=125.0,
            ),
        }
        theo = TheoreticalSpectrum(
            structure_id="struct_test",
            ions=ions,
            structure_mass=250.0,
            monomer_count=2,
            monomer_mass_vector=(125.0, 125.0),
            monomers=monomers,
        )

        res_exp = match_spectrum(
            spec, theo, tolerance=Tolerance(0.02), scoring=ScoringMode.EXP_PEAKS
        )
        assert res_exp is not None
        assert res_exp.matched_peaks_count == 1
        assert len(res_exp.matched_peaks) == 1
        assert res_exp.kakapo_score == 1

    def test_greedy_matching_different_charges(self):
        """Verify that theoretical ions of different charges match different experimental peaks when available."""
        # Experimental peaks at 100.0 and 100.04
        spec = make_spectrum([(100.0, 1.0), (100.04, 1.0)], precursor_mass=250.0, charge=2)
        # Theoretical ions:
        # Ion 1: mz = 100.01 (charge 1) -> within tolerance of 100.0 (diff 0.01) and 100.04 (diff 0.03)
        # Ion 2: mz = 100.03 (charge 2) -> within tolerance of 100.0 (diff 0.03) and 100.04 (diff 0.01)
        # With tolerance 0.02, both ions are within tolerance of both peaks!
        # Because we match sequentially:
        # Ion 1 matches 100.0. Index becomes 1.
        # Ion 2 matches 100.04 (index 1).
        # Both should match!

        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_1",
                    mask=1,
                    rules_shifts=(RuleShift("TEST", 100.01 - PROTON_MASS - 125.0),),
                ),
                TheoreticalIon(
                    fragment_id="frag_2",
                    mask=2,
                    rules_shifts=(RuleShift("TEST", 200.06 - 2 * PROTON_MASS - 125.0),),
                ),
            ]
        )
        monomers = {
            0: Monomer(
                norine_name="Mon_0",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=125.0,
            ),
            1: Monomer(
                norine_name="Mon_1",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=125.0,
            ),
        }
        theo = TheoreticalSpectrum(
            structure_id="struct_test",
            ions=ions,
            structure_mass=250.0,
            monomer_count=2,
            monomer_mass_vector=(125.0, 125.0),
            monomers=monomers,
        )

        res = match_spectrum(spec, theo, tolerance=Tolerance(0.02), scoring=ScoringMode.EXP_PEAKS)
        assert res is not None
        assert res.matched_peaks_count == 2
        assert len(res.matched_peaks) == 2
        assert res.matched_peaks[0].experimental_mz == 100.0
        assert res.matched_peaks[1].experimental_mz == 100.04

    def test_ppm_tolerance_matching(self):
        """Test matching using ppm relative tolerance."""
        # 20 ppm at 100 Da = 0.002 Da.
        # 20 ppm at 1000 Da = 0.02 Da.

        # Test case 1: peak within ppm window
        # mz = 100.0, experimental peak = 100.0015 (diff = 0.0015 <= 0.002)
        spec1 = make_spectrum([(100.0015, 1.0)], precursor_mass=500.0)
        theo1 = make_theoretical_spectrum(
            [100.0 - PROTON_MASS], structure_mass=500.0, monomer_count=1
        )

        # PPM tolerance = 20.0, min_tolerance = 0.0001 (so ppm limit 0.002 takes precedence)
        tol_ppm = Tolerance(value=20.0, unit="ppm", min_tolerance=0.0001)
        res1 = match_spectrum(spec1, theo1, tolerance=tol_ppm, scoring=ScoringMode.EXP_PEAKS)
        assert res1 is not None
        assert res1.matched_peaks_count == 1

        # Test case 2: peak outside ppm window but within min_tolerance
        # mz = 100.0, experimental peak = 100.005 (diff = 0.005 > 0.002)
        # Without min_tolerance it should fail. With min_tolerance = 0.01 it should succeed.
        spec2 = make_spectrum([(100.005, 1.0)], precursor_mass=500.0)

        tol_ppm_low_min = Tolerance(value=20.0, unit="ppm", min_tolerance=0.001)
        res2_fail = match_spectrum(
            spec2, theo1, tolerance=tol_ppm_low_min, scoring=ScoringMode.EXP_PEAKS
        )
        assert res2_fail is not None
        assert res2_fail.matched_peaks_count == 0

        tol_ppm_high_min = Tolerance(value=20.0, unit="ppm", min_tolerance=0.01)
        res2_pass = match_spectrum(
            spec2, theo1, tolerance=tol_ppm_high_min, scoring=ScoringMode.EXP_PEAKS
        )
        assert res2_pass is not None
        assert res2_pass.matched_peaks_count == 1

    def test_intensity_scoring_single_matched_peak(self):
        """Test intensity scoring with a single matched peak."""
        # 1 experimental peak
        peaks = [(100.0, 100.0)]
        # 1 matching theoretical ion (mz = 100.0 at charge 1)
        ion_masses = [100.0 - PROTON_MASS]
        spec = make_spectrum(peaks, precursor_mass=500.0)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.INTENSITY
        )
        assert result is not None
        assert result.matched_peaks_count == 1
        assert result.kakapo_score == pytest.approx(3.846)

    def test_intensity_scoring_multiple_ranked_peaks(self):
        """Test intensity scoring with multiple matched peaks of different ranks."""
        peaks2 = [(100.0, 100.0), (200.0, 50.0)]
        ion_masses2 = [100.0 - PROTON_MASS, 200.0 - PROTON_MASS]
        spec2 = make_spectrum(peaks2, precursor_mass=500.0)
        theo2 = make_theoretical_spectrum(
            ion_masses2, structure_mass=500.0, monomer_count=len(ion_masses2)
        )
        result2 = match_spectrum(
            spec2, theo2, tolerance=Tolerance(0.01), scoring=ScoringMode.INTENSITY
        )
        assert result2 is not None
        assert result2.matched_peaks_count == 2
        assert result2.kakapo_score == pytest.approx(3.846 + 4.059)

    def test_intensity_scoring_priority(self):
        """Test charge priority selection (charge 1 over charge 2) in representative matching."""
        peaks = [(100.0, 100.0)]
        # Two theoretical ions that match 100.0 at charge 1 and charge 2 respectively
        ion_masses = [100.0 - PROTON_MASS, 200.0 - 2 * PROTON_MASS]
        spec = make_spectrum(peaks, precursor_mass=500.0, charge=2)
        theo = make_theoretical_spectrum(
            ion_masses, structure_mass=500.0, monomer_count=len(ion_masses)
        )
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.INTENSITY
        )
        assert result is not None
        assert result.matched_peaks_count == 1
        assert result.kakapo_score == pytest.approx(3.203)
        assert result.matched_peaks[0].charge == 1

    def test_intensity_scoring_missing_group_clustered(self):
        """Test missing peak clustering under 2 * tolerance distance constraint."""
        # Tolerance = 0.02. 2 * tolerance = 0.04.
        tol = Tolerance(0.02)
        # Experimental peak far away so nothing matches
        peaks = [(1000.0, 1.0)]
        spec = make_spectrum(peaks, precursor_mass=1000.0, charge=1)

        # Within 2 * tolerance -> single missing group
        ion_masses_clustered = [300.0 - PROTON_MASS, 300.01 - PROTON_MASS]
        theo_clustered = make_theoretical_spectrum(
            ion_masses_clustered, structure_mass=1000.0, monomer_count=2
        )
        res1 = match_spectrum(spec, theo_clustered, tolerance=tol, scoring=ScoringMode.INTENSITY)
        assert res1 is not None
        assert res1.matched_peaks_count == 0
        assert res1.kakapo_score == pytest.approx(-0.337)

    def test_intensity_scoring_missing_groups_separated(self):
        """Test that missing peaks outside 2 * tolerance are penalized as separate groups."""
        tol = Tolerance(0.02)
        peaks = [(1000.0, 1.0)]
        spec = make_spectrum(peaks, precursor_mass=1000.0, charge=1)

        # Outside 2 * tolerance -> separate missing groups
        ion_masses_separated = [300.0 - PROTON_MASS, 300.1 - PROTON_MASS]
        theo_separated = make_theoretical_spectrum(
            ion_masses_separated, structure_mass=1000.0, monomer_count=2
        )
        res2 = match_spectrum(spec, theo_separated, tolerance=tol, scoring=ScoringMode.INTENSITY)
        assert res2 is not None
        assert res2.matched_peaks_count == 0
        assert res2.kakapo_score == pytest.approx(-0.337 * 2)
