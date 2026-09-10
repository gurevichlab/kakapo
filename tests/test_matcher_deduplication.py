from kakapo.core.matcher.deduplication import _deduplicate_results
from kakapo.models import MatchResult, MonomerModification, Tolerance


class TestSpectrumMatcherDeduplication:
    """Unit tests for MatchResult deduplication and representative selection."""

    def test_deduplicate_results_basic(self) -> None:
        """Verify deduplication marks the correct representative and setsduplicate_representative."""
        # Two results with same spectrum and score, and close theoretical masses.
        r1 = MatchResult(
            structure_id="struct_1",
            spectrum_id="spec_1",
            structure_mass=100.0,
            spectrum_mass=100.5,
            tolerance=Tolerance(0.05),
            matched_peaks_count=5,
            theoretical_peaks_count=10,
            experimental_peaks_count=15,
            monomer_count=3,
            unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
            kakapo_score=0.8,
            post_modification_ion_masses=[100.0, 200.0],
            modifications=(),
        )

        # Duplicate of r1: same score, same spectrum, close masses (within 0.05 tolerance)
        r2 = MatchResult(
            structure_id="struct_2",
            spectrum_id="spec_1",
            structure_mass=100.0,
            spectrum_mass=100.5,
            tolerance=Tolerance(0.05),
            matched_peaks_count=5,
            theoretical_peaks_count=10,
            experimental_peaks_count=15,
            monomer_count=3,
            unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
            kakapo_score=0.8,
            post_modification_ion_masses=[100.01, 200.01],
            modifications=(),
        )

        # Not a duplicate: different score
        r3 = MatchResult(
            structure_id="struct_3",
            spectrum_id="spec_1",
            structure_mass=100.0,
            spectrum_mass=100.5,
            tolerance=Tolerance(0.05),
            matched_peaks_count=5,
            theoretical_peaks_count=10,
            experimental_peaks_count=15,
            monomer_count=3,
            unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
            kakapo_score=0.9,
            post_modification_ion_masses=[100.0, 200.0],
            modifications=(),
        )

        # Not a duplicate: different spectrum
        r4 = MatchResult(
            structure_id="struct_4",
            spectrum_id="spec_2",
            structure_mass=100.0,
            spectrum_mass=100.5,
            tolerance=Tolerance(0.05),
            matched_peaks_count=5,
            theoretical_peaks_count=10,
            experimental_peaks_count=15,
            monomer_count=3,
            unmodified_monomer_mass_vector=(100.0, 100.0, 100.0),
            kakapo_score=0.8,
            post_modification_ion_masses=[100.0, 200.0],
            modifications=(),
        )

        results = [r1, r2, r3, r4]
        deduped = _deduplicate_results(results)

        # Map by structure_id for easy assertion
        deduped_map = {r.structure_id: r for r in deduped}

        # Since r1 and r2 are duplicates:
        # One is representative (duplicate_representative is None), the other points to it.
        # Priority for representative: modifications, structure_id.
        # Both have 0 modifications.
        # "struct_1" < "struct_2", so r1 must be the representative!
        assert deduped_map["struct_1"].duplicate_representative is None
        assert deduped_map["struct_2"].duplicate_representative == deduped_map["struct_1"].result_id

        # r3 and r4 are not duplicates
        assert deduped_map["struct_3"].duplicate_representative is None
        assert deduped_map["struct_4"].duplicate_representative is None

    def test_deduplicate_results_priority(self) -> None:
        """Verify representative selection priority: modifications, structure_id."""
        # All have same score, same spectrum, same masses.
        # r1: 1 modification
        # r2: 0 modifications, structure_id="struct_2"
        # r3: 0 modifications, structure_id="struct_B"
        # r4: 0 modifications, structure_id="struct_A"

        # We expect representative to be r4 (0 mods, "struct_A").

        common_args = {
            "spectrum_id": "spec_1",
            "structure_mass": 100.0,
            "spectrum_mass": 100.5,
            "tolerance": Tolerance(0.05),
            "matched_peaks_count": 5,
            "monomer_count": 3,
            "unmodified_monomer_mass_vector": (100.0, 100.0, 100.0),
            "kakapo_score": 0.8,
            "post_modification_ion_masses": [100.0, 200.0],
            "experimental_peaks_count": 15,
        }

        mod = MonomerModification(
            monomer_idx=0, initial_monomer_name="A", mass_difference=1.0, initial_monomer_mass=1.0
        )

        r1 = MatchResult(
            structure_id="struct_1", modifications=(mod,), theoretical_peaks_count=10, **common_args
        )
        r2 = MatchResult(
            structure_id="struct_C", modifications=(), theoretical_peaks_count=15, **common_args
        )
        r3 = MatchResult(
            structure_id="struct_B", modifications=(), theoretical_peaks_count=10, **common_args
        )
        r4 = MatchResult(
            structure_id="struct_A", modifications=(), theoretical_peaks_count=10, **common_args
        )

        results = [r1, r2, r3, r4]
        deduped = _deduplicate_results(results)

        deduped_map = {r.structure_id: r for r in deduped}

        # r4 must be representative
        assert deduped_map["struct_A"].duplicate_representative is None
        # others must point to r4
        rep_id = deduped_map["struct_A"].result_id
        assert deduped_map["struct_1"].duplicate_representative == rep_id
        assert deduped_map["struct_C"].duplicate_representative == rep_id
        assert deduped_map["struct_B"].duplicate_representative == rep_id

    def test_deduplicate_results_nerpa_score_priority(self) -> None:
        """Verify representative selection priority considers higher nerpa_score before structure_id."""
        common_args = {
            "spectrum_id": "spec_1",
            "structure_mass": 100.0,
            "spectrum_mass": 100.5,
            "tolerance": Tolerance(0.05),
            "matched_peaks_count": 5,
            "monomer_count": 3,
            "unmodified_monomer_mass_vector": (100.0, 100.0, 100.0),
            "kakapo_score": 0.8,
            "post_modification_ion_masses": [100.0, 200.0],
            "experimental_peaks_count": 15,
            "theoretical_peaks_count": 10,
            "modifications": (),
        }

        # r1 has lower structure_id "struct_A", but nerpa_score=5.0
        r1 = MatchResult(structure_id="struct_A", nerpa_score=5.0, **common_args)
        # r2 has higher structure_id "struct_Z", but higher nerpa_score=20.0
        r2 = MatchResult(structure_id="struct_Z", nerpa_score=20.0, **common_args)
        # r3 has nerpa_score=None
        r3 = MatchResult(structure_id="struct_0", nerpa_score=None, **common_args)

        results = [r1, r2, r3]
        deduped = _deduplicate_results(results)

        deduped_map = {r.structure_id: r for r in deduped}

        # r2 ("struct_Z") has highest nerpa_score (20.0), so it must be the representative!
        assert deduped_map["struct_Z"].duplicate_representative is None
        rep_id = deduped_map["struct_Z"].result_id
        assert deduped_map["struct_A"].duplicate_representative == rep_id
        assert deduped_map["struct_0"].duplicate_representative == rep_id
