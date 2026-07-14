"""Unit tests for the main pipeline orchestration in main.py."""

from pathlib import Path

import pytest

from kakapo.constants import FragmentationRule
from kakapo.main import run_pipeline
from kakapo.models import MatchResult, Tolerance


@pytest.fixture
def temp_files(tmp_path: Path):
    """Fixture to create sample MGF and rBAN YAML files for pipeline testing."""
    # 1. Create a sample MGF file
    mgf_content = """BEGIN IONS
TITLE=spec_1
PEPMASS=147.1128
CHARGE=1+
147.1128 100.0
END IONS
BEGIN IONS
TITLE=spec_2
PEPMASS=260.1972
CHARGE=1+
147.1128 50.0
260.1972 100.0
END IONS
"""
    mgf_file = tmp_path / "test_spectra.mgf"
    mgf_file.write_text(mgf_content, encoding="utf-8")

    # 2. Create sample rBAN YAML structures
    # A single alanine molecule
    yaml_ala = """
compound_id: "Ala_structure"
metadata:
  name: "Ala"
  smiles: "CC(N)C(=O)O"
atoms:
  0: {name: "N", hydrogens: 2}
  1: {name: "C", hydrogens: 1}
  2: {name: "C", hydrogens: 0}
  3: {name: "O", hydrogens: 0}
  4: {name: "O", hydrogens: 1}
monomers:
  0: {name: "Ala", atoms: [0, 1, 2, 3, 4], chirality: "L", is_pks_hybrid: false}
monomer_bonds: []
atomic_bonds:
  - - [0, 1]
    - {arity: "1", bond_type: "SINGLE"}
  - - [1, 2]
    - {arity: "1", bond_type: "SINGLE"}
  - - [2, 3]
    - {arity: "2", bond_type: "DOUBLE"}
  - - [2, 4]
    - {arity: "1", bond_type: "SINGLE"}
"""
    yaml_ala_file = tmp_path / "ala.yaml"
    yaml_ala_file.write_text(yaml_ala, encoding="utf-8")

    # An Ala-Ala dimer
    yaml_ala_dimer = """
compound_id: "Ala_Dimer_structure"
metadata:
  name: "Ala_Dimer"
  smiles: "CC(N)C(=O)NC(C)C(=O)O"
atoms:
  0: {name: "N", hydrogens: 2}
  1: {name: "C", hydrogens: 1}
  2: {name: "C", hydrogens: 0}
  3: {name: "O", hydrogens: 0}
  4: {name: "O", hydrogens: 1}
  5: {name: "N", hydrogens: 1}
  6: {name: "C", hydrogens: 1}
  7: {name: "C", hydrogens: 0}
  8: {name: "O", hydrogens: 0}
  9: {name: "O", hydrogens: 1}
monomers:
  0: {name: "Ala", atoms: [0, 1, 2, 3, 4], chirality: "L", is_pks_hybrid: false}
  1: {name: "Ala", atoms: [5, 6, 7, 8, 9], chirality: "L", is_pks_hybrid: false}
monomer_bonds:
  - - [0, 1]
    - - monomer_to_atom: {0: 2, 1: 5}
        atomic_edge: {arity: "1", bond_type: "AMINO"}
atomic_bonds:
  - - [0, 1]
    - {arity: "1", bond_type: "SINGLE"}
  - - [1, 2]
    - {arity: "1", bond_type: "SINGLE"}
  - - [2, 3]
    - {arity: "2", bond_type: "DOUBLE"}
  - - [2, 4]
    - {arity: "1", bond_type: "SINGLE"}
  - - [5, 6]
    - {arity: "1", bond_type: "SINGLE"}
  - - [6, 7]
    - {arity: "1", bond_type: "SINGLE"}
  - - [7, 8]
    - {arity: "2", bond_type: "DOUBLE"}
  - - [7, 9]
    - {arity: "1", bond_type: "SINGLE"}
  - - [2, 5]
    - {arity: "1", bond_type: "AMINO"}
"""
    yaml_dimer_file = tmp_path / "dimer.yaml"
    yaml_dimer_file.write_text(yaml_ala_dimer, encoding="utf-8")

    return {
        "mgf": mgf_file,
        "ala_yaml": yaml_ala_file,
        "dimer_yaml": yaml_dimer_file,
        "tmp_path": tmp_path,
    }


class TestPipelineOrchestration:
    """Test suite for the Kakapo workflow pipeline run_pipeline function."""

    def test_run_pipeline_single_files(self, temp_files):
        """Test running pipeline with single paths for spectra and structures."""
        results, molecules, spectra = run_pipeline(
            spectra_files=temp_files["mgf"],
            structure_files=temp_files["ala_yaml"],
            tolerance=Tolerance(0.5),
            apply_filters=False,
            min_peaks=1,
            rules=None,
            use_cache=False,
            cache_dir=temp_files["tmp_path"] / ".kakapo_cache",
        )

        assert isinstance(results, list)
        assert isinstance(molecules, list)
        assert len(molecules) == 1
        assert molecules[0].structure_id == "Ala_structure"
        assert isinstance(spectra, list)
        assert len(spectra) == 2
        # Should match each of the 2 spectra to the 1 structure -> 2 results
        assert len(results) == 2
        for r in results:
            assert isinstance(r, MatchResult)
            assert r.structure_id == "Ala_structure"

    def test_run_pipeline_multiple_files(self, temp_files):
        """Test running pipeline with lists of paths for multiple structures."""
        results, molecules, spectra = run_pipeline(
            spectra_files=[temp_files["mgf"]],
            structure_files=[temp_files["ala_yaml"], temp_files["dimer_yaml"]],
            tolerance=Tolerance(0.5),
            apply_filters=False,
            min_peaks=1,
            rules=None,
            use_cache=False,
            cache_dir=temp_files["tmp_path"] / ".kakapo_cache",
        )

        assert isinstance(spectra, list)

        # 2 spectra * 2 structures = 4 combinations, but only 2 (with Ala_structure) have
        # matching peaks > 0
        assert len(results) == 2
        assert len(molecules) == 2
        molecule_ids = {m.structure_id for m in molecules}
        assert molecule_ids == {"Ala_structure", "Ala_Dimer_structure"}

    def test_run_pipeline_sorting_order(self, temp_files):
        """Test sorting of results: descending matched peaks, then ascending IDs."""
        results, _, _ = run_pipeline(
            spectra_files=temp_files["mgf"],
            structure_files=[temp_files["ala_yaml"], temp_files["dimer_yaml"]],
            tolerance=Tolerance(0.5),
            apply_filters=False,
            min_peaks=1,
            rules=None,
            use_cache=False,
            cache_dir=temp_files["tmp_path"] / ".kakapo_cache",
        )

        # Ensure they are sorted primarily by matched_peaks_count in descending order.
        # Secondarily by spectrum_id ascending, and thirdly by structure_id ascending.
        counts = [r.matched_peaks_count for r in results]
        assert counts == sorted(counts, reverse=True)

        for i in range(len(results) - 1):
            if results[i].matched_peaks_count == results[i + 1].matched_peaks_count:
                if results[i].spectrum_id == results[i + 1].spectrum_id:
                    assert results[i].structure_id <= results[i + 1].structure_id
                else:
                    assert results[i].spectrum_id <= results[i + 1].spectrum_id

    def test_run_pipeline_custom_rules(self, temp_files):
        """Test passing custom fragmentation rules."""
        # 1. Create a custom MGF file containing peaks matching AMINO cleavage fragments of the dimer
        # Dimer structure mass is ~147.0405. Cleaved N-side: 75.0242 (m/z 76.0315), C-side: 72.0163 (m/z 73.0236)
        custom_mgf_content = """BEGIN IONS
TITLE=custom_spec
PEPMASS=148.0478
CHARGE=1+
76.0315 100.0
73.0236 100.0
END IONS
"""
        custom_mgf_file = temp_files["tmp_path"] / "custom_spectra.mgf"
        custom_mgf_file.write_text(custom_mgf_content, encoding="utf-8")

        custom_rules = (FragmentationRule("AMINO", 1.0, -1.0),)

        # Match with custom rules (AMINO is breakable)
        results, _, _ = run_pipeline(
            spectra_files=custom_mgf_file,
            structure_files=temp_files["dimer_yaml"],
            tolerance=Tolerance(0.5),
            apply_filters=False,
            min_peaks=1,
            rules=custom_rules,
            use_cache=False,
            cache_dir=temp_files["tmp_path"] / ".kakapo_cache",
        )

        assert len(results) == 1
        assert results[0].matched_peaks_count == 2
        assert results[0].structure_id == "Ala_Dimer_structure"

        # Match with default rules (AMINO is NOT breakable, yields 0 matches and gets excluded)
        default_results, _, _ = run_pipeline(
            spectra_files=custom_mgf_file,
            structure_files=temp_files["dimer_yaml"],
            tolerance=Tolerance(0.5),
            apply_filters=False,
            min_peaks=1,
            rules=None,
            use_cache=False,
            cache_dir=temp_files["tmp_path"] / ".kakapo_cache",
        )
        assert len(default_results) == 0

    def test_run_pipeline_missing_inputs(self, tmp_path):
        """Test raising ValueError for empty spectra/structure file lists."""
        with pytest.raises(ValueError, match="No spectra files"):
            run_pipeline(
                spectra_files=[],
                structure_files=["dummy.yaml"],
                tolerance=Tolerance(0.1),
                apply_filters=True,
                min_peaks=1,
                rules=None,
                use_cache=False,
                cache_dir=tmp_path / ".kakapo_cache",
            )

        with pytest.raises(ValueError, match="No structure files"):
            run_pipeline(
                spectra_files=["dummy.mgf"],
                structure_files=[],
                tolerance=Tolerance(0.1),
                apply_filters=True,
                min_peaks=1,
                rules=None,
                use_cache=False,
                cache_dir=tmp_path / ".kakapo_cache",
            )

    def test_run_pipeline_nonexistent_files(self, tmp_path):
        """Test raising FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            run_pipeline(
                spectra_files="nonexistent.mgf",
                structure_files="nonexistent.yaml",
                tolerance=Tolerance(0.1),
                apply_filters=True,
                min_peaks=1,
                rules=None,
                use_cache=False,
                cache_dir=tmp_path / ".kakapo_cache",
            )
