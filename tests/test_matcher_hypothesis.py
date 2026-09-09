import numpy as np
import pytest

from kakapo.constants import PROTON_MASS
from kakapo.core.matcher.orchestrator import match_spectrum
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


class TestSpectrumMatcherHypothesis:
    """Unit tests for the spectrum matching modification hypothesis generation and evaluation."""

    @pytest.mark.parametrize(
        "monomer_masses,precursor_mass,peaks,expected_matched,expected_mod,expected_idx",
        [
            # Positive shift (monomer 1 modified: 150.0 + 30.0 = 180.0)
            (
                (100.0, 150.0, 200.0),
                480.0,
                [(100.0 + 1.007276, 1.0), (180.0 + 1.007276, 1.0), (200.0 + 1.007276, 1.0)],
                3,
                30.0,
                1,
            ),
            # Negative shift (monomer 2 modified: 200.0 - 20.0 = 180.0)
            (
                (100.0, 150.0, 200.0),
                430.0,
                [(100.0 + 1.007276, 1.0), (150.0 + 1.007276, 1.0), (180.0 + 1.007276, 1.0)],
                3,
                -20.0,
                2,
            ),
            # Zero shift (no modifications, modified_monomer_idx is None)
            (
                (100.0, 150.0, 200.0),
                450.0,
                [(100.0 + 1.007276, 1.0), (150.0 + 1.007276, 1.0), (200.0 + 1.007276, 1.0)],
                3,
                0.0,
                None,
            ),
            # Single monomer
            (
                (150.0,),
                160.0,
                [(160.0 + 1.007276, 1.0)],
                1,
                10.0,
                0,
            ),
            # Non-positive monomer skipped (monomer 0 becomes 15 - 20 = -5 <= 0, monomer 1 modified:
            # 150 - 20 = 130)
            (
                (15.0, 150.0),
                145.0,
                [(15.0 + 1.007276, 1.0), (130.0 + 1.007276, 1.0), (145.0 + 1.007276, 1.0)],
                3,
                -20.0,
                1,
            ),
            # Non-positive monomer NOT skipped if within tolerance (monomer 0 becomes 0.005 - 0.012 = -0.007 >= -0.01)
            (
                (0.005, 150.0),
                149.993,
                [(150.0 + 1.007276, 1.0), (149.993 + 1.007276, 1.0)],
                2,
                -0.012,
                0,
            ),
        ],
    )
    def test_modification_tolerant_matching_scenarios(
        self, monomer_masses, precursor_mass, peaks, expected_matched, expected_mod, expected_idx
    ):
        """Test various modification-tolerant matching scenarios including shift types and edge cases."""
        ions = frozenset(
            TheoreticalIon(fragment_id=f"frag_{idx}", mask=1 << idx)
            for idx in range(len(monomer_masses))
        )
        if len(monomer_masses) == 2:
            ions = frozenset(list(ions) + [TheoreticalIon(fragment_id="frag_2", mask=3)])

        monomers = {
            idx: Monomer(
                norine_name=f"Mon_{idx}",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=mass,
            )
            for idx, mass in enumerate(monomer_masses)
        }
        theo = TheoreticalSpectrum(
            structure_id="struct_mod",
            ions=ions,
            structure_mass=sum(monomer_masses),
            monomer_count=len(monomer_masses),
            monomer_mass_vector=monomer_masses,
            monomers=monomers,
        )

        spec = make_spectrum(peaks, precursor_mass=precursor_mass, charge=1)
        result = match_spectrum(
            spec, theo, tolerance=Tolerance(0.01), scoring=ScoringMode.EXP_PEAKS
        )
        assert result is not None
        assert result.matched_peaks_count == expected_matched
        if expected_idx is None:
            assert len(result.modifications) == 0
        else:
            assert len(result.modifications) == 1
            mod = result.modifications[0]
            assert mod.mass_difference == pytest.approx(expected_mod)
            assert mod.monomer_idx == expected_idx
            assert mod.initial_monomer_mass == monomer_masses[expected_idx]

    def test_x_mode_matching_unambiguous(self) -> None:
        """Verify that x_mode=True only permits modifications on monomers starting with 'X'."""
        monomers = {
            0: Monomer(
                norine_name="X0",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
            1: Monomer(
                norine_name="Ala",
                residue="A",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
        }

        # frag_0 has a shift of +5.0 (mass 105.0 unmodified)
        # frag_1 has a shift of 0.0 (mass 100.0 unmodified)
        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_0", mask=1, rules_shifts=(RuleShift("PEPTIDE", 5.0),)
                ),
                TheoreticalIon(fragment_id="frag_1", mask=2),
            ]
        )

        theo = TheoreticalSpectrum(
            structure_id="struct_x",
            ions=ions,
            structure_mass=205.0,  # sum of unmodified monomers (200.0) + shift (5.0)
            monomer_count=2,
            monomer_mass_vector=(100.0, 100.0),
            monomers=monomers,
        )

        # Precursor neutral mass = 215.0 (+10.0 mod)
        # If we modify X0 (idx 0), we expect:
        # frag_0 -> 115.0
        # frag_1 -> 100.0
        #
        # If we modify Ala (idx 1), we expect:
        # frag_0 -> 105.0
        # frag_1 -> 110.0

        # Let's test experimental spectrum that has modified Ala (110.0)
        # This requires modifying Ala (idx 1) to match.
        spec_ala = Spectrum(
            spectrum_id="spec_ala",
            source_file_path="test.mgf",
            mzs=np.array([110.0 + 1.0078]),
            intensities=np.array([1.0]),
            precursor_mass=215.0 + 1.0078,
            charge=1,
        )

        # In FREE_MOD mode, it should successfully match (modifying Ala, idx 1, to get peak 110.0)
        res_normal = match_spectrum(
            spec_ala,
            theo,
            tolerance=Tolerance(0.02),
            scoring=ScoringMode.EXP_PEAKS,
            mode=ModificationMode.FREE_MOD,
        )
        assert res_normal is not None
        assert res_normal.matched_peaks_count == 1
        assert len(res_normal.modifications) == 1
        assert res_normal.modifications[0].monomer_idx == 1

        # In X_MOD mode, it should NOT match since Ala doesn't start with X,
        # and modifying X0 (idx 0) doesn't produce peak 110.0 (it produces 115.0 and 100.0)
        res_x = match_spectrum(
            spec_ala,
            theo,
            tolerance=Tolerance(0.02),
            scoring=ScoringMode.EXP_PEAKS,
            mode=ModificationMode.X_MOD,
        )
        assert res_x is not None
        assert res_x.matched_peaks_count == 0
        assert len(res_x.modifications) == 0

    def test_no_mod_mode(self) -> None:
        """Verify that NO_MOD mode skips modification search and returns 0 matches when precursor mass differs."""
        monomers = {
            0: Monomer(
                norine_name="Ala",
                residue="A",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
        }
        ions = frozenset([TheoreticalIon(fragment_id="frag_0", mask=1)])
        theo = TheoreticalSpectrum(
            structure_id="struct_nomod",
            ions=ions,
            structure_mass=100.0,
            monomer_count=1,
            monomer_mass_vector=(100.0,),
            monomers=monomers,
        )

        # Spectrum with precursor mass = 120.0 (mass_mod = +20.0 > tolerance)
        spec = Spectrum(
            spectrum_id="spec_nomod",
            source_file_path="test.mgf",
            mzs=np.array([120.0 + 1.0078]),
            intensities=np.array([1.0]),
            precursor_mass=120.0 + 1.0078,
            charge=1,
        )

        res_no_mod = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.02),
            scoring=ScoringMode.EXP_PEAKS,
            mode=ModificationMode.NO_MOD,
        )
        assert res_no_mod is None

        res_free_mod = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.02),
            scoring=ScoringMode.EXP_PEAKS,
            mode=ModificationMode.FREE_MOD,
        )
        assert res_free_mod is not None
        assert res_free_mod.matched_peaks_count == 1
        assert len(res_free_mod.modifications) == 1

    def test_two_mod_matching_success(self) -> None:
        """Test two-modification matching with a known base monomer modification and X monomer residual shift."""
        monomers = {
            0: Monomer(
                norine_name="Ala",
                residue="A",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
            1: Monomer(
                norine_name="Xle",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=120.0,
            ),
        }

        # Unmodified structure mass = 220.0
        # Known mod on Ala: NAc-Ala (+C2H2O, mass_difference = 42.010565)
        # Expected Ala mass = 142.010565
        # Desired residual shift on Xle = +20.0 -> Xle mass = 140.0
        # Precursor neutral mass = 220.0 + 42.010565 + 20.0 = 282.010565

        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_0", mask=1, rules_shifts=(RuleShift("PEPTIDE", 0.0),)
                ),
                TheoreticalIon(
                    fragment_id="frag_1", mask=2, rules_shifts=(RuleShift("PEPTIDE", 0.0),)
                ),
            ]
        )

        theo = TheoreticalSpectrum(
            structure_id="struct_two_mod",
            ions=ions,
            structure_mass=220.0,
            monomer_count=2,
            monomer_mass_vector=(100.0, 120.0),
            monomers=monomers,
        )

        spec = Spectrum(
            spectrum_id="spec_two_mod",
            source_file_path="test.mgf",
            mzs=np.array([140.0 + 1.007276, 142.010565 + 1.007276]),
            intensities=np.array([100.0, 100.0]),
            precursor_mass=282.010565 + 1.007276,
            charge=1,
        )

        result = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            mode=ModificationMode.TWO_MOD,
        )

        assert result is not None
        assert result.matched_peaks_count == 2
        assert len(result.modifications) == 2

        mod1 = result.modifications[0]
        assert mod1.monomer_idx == 0
        assert mod1.initial_monomer_name == "Ala"
        assert mod1.mass_difference == pytest.approx(42.010565)
        assert mod1.new_monomer_name == "NAc-Ala"
        assert mod1.formula_difference == "+C2H2O"
        assert mod1.initial_monomer_mass == 100.0

        mod2 = result.modifications[1]
        assert mod2.monomer_idx == 1
        assert mod2.initial_monomer_name == "Xle"
        assert mod2.mass_difference == pytest.approx(20.0)
        assert mod2.new_monomer_name is None
        assert mod2.initial_monomer_mass == 120.0

    def test_two_mod_matching_no_x_monomers(self) -> None:
        """Verify that two-modification mode produces 0 matches if no X monomers exist in structure."""
        monomers = {
            0: Monomer(
                norine_name="Ala",
                residue="A",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
            1: Monomer(
                norine_name="Val",
                residue="V",
                is_methylated=False,
                chirality=Chirality.L,
                mass=120.0,
            ),
        }

        ions = frozenset([TheoreticalIon(fragment_id="frag_0", mask=1)])
        theo = TheoreticalSpectrum(
            structure_id="struct_no_x",
            ions=ions,
            structure_mass=220.0,
            monomer_count=2,
            monomer_mass_vector=(100.0, 120.0),
            monomers=monomers,
        )

        spec = Spectrum(
            spectrum_id="spec_test",
            source_file_path="test.mgf",
            mzs=np.array([150.0]),
            intensities=np.array([10.0]),
            precursor_mass=280.0,
            charge=1,
        )

        result = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            mode=ModificationMode.TWO_MOD,
        )

        assert result is None

    def test_two_mod_matching_includes_single_mod(self) -> None:
        """Verify that TWO_MOD mode can match a single-modification hypothesis if optimal."""
        monomers = {
            0: Monomer(
                norine_name="Xle",
                residue="X",
                is_methylated=False,
                chirality=Chirality.L,
                mass=100.0,
            ),
            1: Monomer(
                norine_name="Val",
                residue="V",
                is_methylated=False,
                chirality=Chirality.L,
                mass=120.0,
            ),
        }

        ions = frozenset(
            [
                TheoreticalIon(
                    fragment_id="frag_0", mask=1, rules_shifts=(RuleShift("PEPTIDE", 0.0),)
                )
            ]
        )
        theo = TheoreticalSpectrum(
            structure_id="struct_single_mod_in_two_mod",
            ions=ions,
            structure_mass=220.0,
            monomer_count=2,
            monomer_mass_vector=(100.0, 120.0),
            monomers=monomers,
        )

        # Precursor neutral mass = 220.0 + 50.0 = 270.0
        # Peak matching Ala (+50.0) + proton = 150.0 + 1.007276
        spec = Spectrum(
            spectrum_id="spec_test",
            source_file_path="test.mgf",
            mzs=np.array([150.0 + PROTON_MASS]),
            intensities=np.array([100.0]),
            precursor_mass=270.0 + PROTON_MASS,
            charge=1,
        )

        result = match_spectrum(
            spec,
            theo,
            tolerance=Tolerance(0.01),
            scoring=ScoringMode.EXP_PEAKS,
            mode=ModificationMode.TWO_MOD,
        )

        assert result is not None
        assert result.matched_peaks_count == 1
        assert len(result.modifications) == 1
        assert result.modifications[0].monomer_idx == 0
        assert result.modifications[0].mass_difference == pytest.approx(50.0)
