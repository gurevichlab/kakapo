"""Unit tests for the spectrum match results writer module."""

import csv
import json
from pathlib import Path

import pytest

from kakapo.io.writer import (
    format_as_csv,
    format_as_json,
    format_as_table,
    write_results,
    write_spectra,
    write_structures,
)
from kakapo.models import (
    Chirality,
    MatchedPeak,
    MatchResult,
    Molecule,
    Monomer,
    MonomerModification,
    MonomerOrigin,
    NerpaMSData,
    Peak,
    Spectrum,
    Tolerance,
)

DUMMY_SPEC = Spectrum(
    spectrum_id="spec_1",
    source_file_path="test_file.mgf",
    peaks=frozenset([Peak(100.0, 1.0)]),
    precursor_mass=500.0,
)


@pytest.fixture
def sample_results() -> list[MatchResult]:
    """Provide a standard list of MatchResult objects for testing."""
    return [
        MatchResult(
            structure_id="NRP_001",
            spectrum_id="spec_101",
            structure_mass=800.45,
            spectrum_mass=800.46,
            tolerance=Tolerance(0.02),
            score=12.3456,
            matched_peaks_count=5,
            theoretical_peaks_count=10,
            experimental_peaks_count=15,
            monomer_count=7,
            unmodified_monomer_mass_vector=(110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0),
            matched_peaks=[],
            modifications=(
                MonomerModification(
                    monomer_idx=0,
                    initial_monomer_name="Ala",
                    mass_difference=0.01,
                    new_monomer_name=None,
                    formula_difference=None,
                    initial_monomer_mass=110.0,
                ),
            ),
        ),
        MatchResult(
            structure_id="NRP_002",
            spectrum_id="spec_102",
            structure_mass=950.50,
            spectrum_mass=950.48,
            tolerance=Tolerance(0.05),
            score=0.0,
            matched_peaks_count=3,
            theoretical_peaks_count=12,
            experimental_peaks_count=20,
            monomer_count=8,
            unmodified_monomer_mass_vector=(110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0),
            matched_peaks=[],
            modifications=(
                MonomerModification(
                    monomer_idx=3,
                    initial_monomer_name="Val",
                    mass_difference=-0.02,
                    new_monomer_name=None,
                    formula_difference=None,
                    initial_monomer_mass=120.0,
                ),
            ),
        ),
    ]


class TestWriterCsvFormatting:
    """Test suite for CSV formatting function."""

    def test_format_as_csv_empty(self) -> None:
        """Test formatting an empty list of results."""
        csv_str = format_as_csv([])
        lines = csv_str.strip().split("\n")
        assert len(lines) == 1  # Only the header
        assert "structure_id" in lines[0]

    def test_format_as_csv_values(self, sample_results: list[MatchResult]) -> None:
        """Test formatting a list of results and parsing back the values."""
        csv_str = format_as_csv(sample_results)
        reader = csv.DictReader(csv_str.splitlines())
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["structure_id"] == "NRP_001"
        assert rows[0]["spectrum_id"] == "spec_101"
        assert rows[0]["structure_mass"] == "800.45000"
        assert rows[1]["structure_mass"] == "950.50000"
        assert rows[0]["tolerance"] == "0.02da"
        assert rows[0]["score"] == "12.346"
        assert int(rows[0]["matched_peaks_count"]) == 5
        assert int(rows[0]["experimental_peaks_count"]) == 15
        assert "Ala[0]" in rows[0]["modifications"]

        assert rows[1]["structure_id"] == "NRP_002"
        assert rows[1]["spectrum_id"] == "spec_102"
        assert rows[1]["score"] == "0.000"
        assert float(rows[1]["spectrum_mass"]) == 950.48
        assert int(rows[1]["experimental_peaks_count"]) == 20
        assert int(rows[1]["monomer_count"]) == 8
        assert "Val[3]" in rows[1]["modifications"]

    def test_format_with_known_modifications(self) -> None:
        """Test formatting results with known modification names and chemical formulas."""
        mod_known = MonomerModification(
            monomer_idx=1,
            initial_monomer_name="Ala",
            initial_monomer_mass=100.0,
            mass_difference=42.010565,
            new_monomer_name="NAc-Ala",
            formula_difference="+C2H2O",
        )
        mod_x = MonomerModification(
            monomer_idx=2,
            initial_monomer_name="Xle",
            initial_monomer_mass=120.0,
            mass_difference=15.0,
            new_monomer_name=None,
            formula_difference=None,
        )
        result = MatchResult(
            structure_id="NRP_MOD",
            spectrum_id="SPEC_MOD",
            structure_mass=500.0,
            spectrum_mass=557.010565,
            tolerance=Tolerance(0.02),
            score=10.0,
            matched_peaks_count=4,
            theoretical_peaks_count=8,
            experimental_peaks_count=12,
            monomer_count=3,
            unmodified_monomer_mass_vector=(100.0, 100.0, 120.0),
            matched_peaks=[],
            modifications=(mod_known, mod_x),
        )

        csv_str = format_as_csv([result])
        assert "Ala[1]->NAc-Ala (+42.0106, +C2H2O)" in csv_str
        assert "Xle[2] (+15.0000)" in csv_str

        table_str = format_as_table([result])
        assert "Ala[1]->NAc-Ala (+42.0106, +C2H2O)" in table_str

        json_str = format_as_json([result])
        data = json.loads(json_str)
        mods = data[0]["modifications"]
        assert len(mods) == 2
        assert mods[0]["new_monomer_name"] == "NAc-Ala"
        assert mods[0]["formula_difference"] == "+C2H2O"
        assert mods[0]["initial_monomer_mass"] == 100.0
        assert mods[1]["monomer_idx"] == 2
        assert mods[1]["initial_monomer_mass"] == 120.0


class TestWriterTableFormatting:
    """Test suite for ASCII table formatting function."""

    def test_format_as_table_empty(self) -> None:
        """Test formatting an empty list of results."""
        table_str = format_as_table([])
        assert "Structure ID" in table_str
        assert "Spectrum ID" in table_str

    def test_format_as_table_values(self, sample_results: list[MatchResult]) -> None:
        """Test formatting results as table contains IDs and numbers."""
        table_str = format_as_table(sample_results)
        assert "NRP_001" in table_str
        assert "spec_101" in table_str
        assert "800.45000" in table_str
        assert "NRP_002" in table_str
        assert "spec_102" in table_str
        assert "950.50000" in table_str
        # Check table headers are present
        assert "Structure ID" in table_str
        assert "Experimental Peaks" in table_str
        assert "Monomer Count" in table_str
        assert "Modifications" in table_str
        assert "Ala[0]" in table_str


class TestWriterJsonFormatting:
    """Test suite for JSON formatting function."""

    def test_format_as_json_empty(self) -> None:
        """Test formatting an empty list of results."""
        json_str = format_as_json([])
        assert json_str.strip() == "[]"

    def test_format_as_json_values(self, sample_results: list[MatchResult]) -> None:
        """Test formatting a list of results and parsing back the values."""
        # Let's add some peaks to NRP_001 matched_peaks for mask testing
        sample_results[0] = MatchResult(
            structure_id=sample_results[0].structure_id,
            spectrum_id=sample_results[0].spectrum_id,
            structure_mass=sample_results[0].structure_mass,
            spectrum_mass=sample_results[0].spectrum_mass,
            tolerance=sample_results[0].tolerance,
            score=sample_results[0].score,
            matched_peaks_count=sample_results[0].matched_peaks_count,
            theoretical_peaks_count=sample_results[0].theoretical_peaks_count,
            experimental_peaks_count=sample_results[0].experimental_peaks_count,
            monomer_count=sample_results[0].monomer_count,
            unmodified_monomer_mass_vector=sample_results[0].unmodified_monomer_mass_vector,
            matched_peaks=[
                MatchedPeak(
                    theoretical_mz=100.0,
                    experimental_mz=100.1,
                    experimental_peak_idx=2,
                    charge=1,
                    theoretical_fragment_mask=5,
                )
            ],
            modifications=sample_results[0].modifications,
        )

        json_str = format_as_json(sample_results)
        data = json.loads(json_str)

        assert len(data) == 2
        assert data[0]["structure_id"] == "NRP_001"
        assert data[0]["spectrum_id"] == "spec_101"
        assert data[0]["structure_mass"] == 800.45
        assert data[1]["structure_mass"] == 950.5
        assert data[0]["tolerance"] == "0.02da"
        assert data[0]["score"] == 12.346

        # Test reversed binary mask format (mask = 5 -> "101", padded to monomer_count = 7)
        assert data[0]["matched_peaks"][0]["theoretical_fragment_mask"] == "1010000"
        assert data[0]["matched_peaks"][0]["experimental_peak_idx"] == 2
        assert data[0]["modifications"][0]["initial_monomer_name"] == "Ala"

        assert data[1]["structure_id"] == "NRP_002"
        assert data[1]["score"] == 0.0
        assert data[1]["monomer_count"] == 8
        assert data[1]["modifications"][0]["monomer_idx"] == 3

        assert data[1]["monomer_count"] == 8
        assert data[1]["unmodified_monomer_mass_vector"] == [
            110.0,
            120.0,
            130.0,
            140.0,
            150.0,
            160.0,
            170.0,
            180.0,
        ]


class TestWriterWriteResults:
    """Test suite for the main write_results dispatch function."""

    def test_write_results_stdout_table(
        self, sample_results: list[MatchResult], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test writing table formatted results to stdout."""
        write_results(sample_results, output_path=None, format_type="table")
        captured = capsys.readouterr()
        assert "NRP_001" in captured.out
        assert "Structure ID" in captured.out

    def test_write_results_stdout_csv(
        self, sample_results: list[MatchResult], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Test writing CSV formatted results to stdout using dash parameter."""
        write_results(sample_results, output_path="-", format_type="csv")
        captured = capsys.readouterr()
        assert "structure_id" in captured.out
        assert "NRP_001" in captured.out

    def test_write_results_to_file(self, sample_results: list[MatchResult], tmp_path: Path) -> None:
        """Test writing results to a file path."""
        out_file = tmp_path / "results.csv"
        write_results(sample_results, output_path=out_file, format_type="csv")

        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "structure_id" in content
        assert "NRP_002" in content

    def test_write_results_to_file_creates_directories(
        self, sample_results: list[MatchResult], tmp_path: Path
    ) -> None:
        """Test that missing parent directories are automatically created."""
        out_file = tmp_path / "deep" / "nested" / "output.txt"
        write_results(sample_results, output_path=out_file, format_type="table")

        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "Structure ID" in content
        assert "NRP_001" in content

    def test_write_results_unsupported_format(self, sample_results: list[MatchResult]) -> None:
        """Test that unsupported formats raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported format type"):
            write_results(sample_results, output_path=None, format_type="xml")

    def test_write_results_os_error(
        self, sample_results: list[MatchResult], tmp_path: Path
    ) -> None:
        """Test that OSError propagates from file write failure."""
        # Point to a directory name as the output file to trigger an OSError (IsADirectoryError)
        invalid_path = tmp_path / "invalid_dir"
        invalid_path.mkdir()

        with pytest.raises(OSError):
            write_results(sample_results, output_path=invalid_path, format_type="csv")


class TestWriterWriteStructures:
    """Test suite for the write_structures function."""

    def test_write_structures_stdout_or_none(self) -> None:
        """Test write_structures returns early with no files written for stdout/None."""
        monomer = Monomer(
            norine_name="Ala",
            residue="Ala",
            is_methylated=False,
            chirality=Chirality.L,
        )
        dummy_mol = Molecule(
            structure_id="Ala_structure",
            monomers={0: monomer},
            bonds=(),
        )

        # Should do nothing and not raise any error
        write_structures([dummy_mol], output_path=None)
        write_structures([dummy_mol], output_path="-")

    def test_write_structures_to_directory(self, tmp_path: Path) -> None:
        """Test writing structures when output_path is a directory."""
        monomer = Monomer(
            norine_name="Ala",
            residue="Ala",
            is_methylated=False,
            chirality=Chirality.L,
        )
        dummy_mol = Molecule(
            structure_id="Ala_structure",
            monomers={0: monomer},
            bonds=(),
        )

        # Case 1: output_path is a directory
        write_structures([dummy_mol], output_path=tmp_path)
        expected_file = tmp_path / "structures.json"
        assert expected_file.exists()

        data = json.loads(expected_file.read_text(encoding="utf-8"))
        assert "Ala_structure" in data
        assert "structure_id" not in data["Ala_structure"]
        assert data["Ala_structure"]["input_compound_id"] == "Ala_structure"

    def test_write_structures_to_file_path(self, tmp_path: Path) -> None:
        """Test writing structures when output_path is a specific file path."""
        monomer = Monomer(
            norine_name="Ala",
            residue="Ala",
            is_methylated=False,
            chirality=Chirality.L,
        )
        dummy_mol = Molecule(
            structure_id="Ala_structure",
            monomers={0: monomer},
            bonds=(),
        )

        out_file = tmp_path / "subdir" / "results.csv"
        write_structures([dummy_mol], output_path=out_file)
        expected_file = tmp_path / "subdir" / "structures.json"
        assert expected_file.exists()

        data = json.loads(expected_file.read_text(encoding="utf-8"))
        assert "Ala_structure" in data
        assert "structure_id" not in data["Ala_structure"]
        assert data["Ala_structure"]["input_compound_id"] == "Ala_structure"


class TestWriterWriteSpectra:
    """Test suite for write_spectra function in writer.py."""

    def test_write_spectra_stdout_skips(self) -> None:
        """Test that write_spectra does not write files when stdout is specified."""
        # This shouldn't raise any errors or write files
        write_spectra([DUMMY_SPEC], output_path=None)
        write_spectra([DUMMY_SPEC], output_path="-")

    def test_write_spectra_to_directory(self, tmp_path: Path) -> None:
        """Test writing spectra when output_path is a directory."""
        write_spectra([DUMMY_SPEC], output_path=tmp_path)
        expected_file = tmp_path / "spectra.json"
        assert expected_file.exists()

        data = json.loads(expected_file.read_text(encoding="utf-8"))
        assert "spec_1" in data
        assert data["spec_1"]["peaks"][0]["mass"] == 100.0

    def test_write_spectra_to_file_path(self, tmp_path: Path) -> None:
        """Test writing spectra when output_path is a specific file path."""
        out_file = tmp_path / "subdir" / "results.csv"
        write_spectra([DUMMY_SPEC], output_path=out_file)
        expected_file = tmp_path / "subdir" / "spectra.json"
        assert expected_file.exists()

        data = json.loads(expected_file.read_text(encoding="utf-8"))
        assert "spec_1" in data


class TestWriterNerpaMSDataFormatting:
    """Test suite for NERPA-MS data formatting in CSV, ASCII table, JSON, and structures."""

    def _create_sample_nerpa_data(self) -> tuple[MatchResult, dict[str, Molecule]]:
        nerpa_data = NerpaMSData(
            rank=1,
            score=45.678,
            old_to_new_mon_map=((0, 0), (1, 2)),
            monomer_origins=(
                MonomerOrigin(compound_id="comp_1", monomer_name="D-Ala", monomer_idx=2),
            ),
        )
        monomer = Monomer(
            norine_name="Ala",
            residue="Ala",
            is_methylated=False,
            chirality=Chirality.L,
        )
        mol = Molecule(
            structure_id="NRP_NERPA",
            monomers={0: monomer},
            bonds=(),
            nerpa_ms_data=nerpa_data,
        )
        res = MatchResult(
            structure_id="NRP_NERPA",
            spectrum_id="SPEC_NERPA",
            structure_mass=500.0,
            spectrum_mass=500.0,
            tolerance=Tolerance(0.02),
            score=10.0,
            matched_peaks_count=4,
            theoretical_peaks_count=8,
            experimental_peaks_count=12,
            monomer_count=3,
            unmodified_monomer_mass_vector=(100.0, 100.0, 120.0),
            matched_peaks=[],
            modifications=(),
        )
        return res, {mol.structure_id: mol}

    def test_format_csv_with_nerpa_ms_data(self) -> None:
        """Test formatting MatchResult into CSV with nerpa_ms_data present."""
        res, mol_dict = self._create_sample_nerpa_data()
        csv_str = format_as_csv([res], molecules=mol_dict)
        assert "nerpa_score,nerpa_rank,nerpa_modifications_count" in csv_str
        assert "45.678,1,2" in csv_str

    def test_format_table_with_nerpa_ms_data(self) -> None:
        """Test formatting MatchResult into ASCII table with nerpa_ms_data present."""
        res, mol_dict = self._create_sample_nerpa_data()
        table_str = format_as_table([res], molecules=mol_dict)
        assert "NERPA Score" in table_str
        assert "NERPA Rank" in table_str
        assert "NERPA Modifications" in table_str
        assert "45.678" in table_str
        assert "1" in table_str
        assert "2" in table_str

    def test_format_json_with_nerpa_ms_data(self) -> None:
        """Test formatting MatchResult into JSON with nerpa_ms_data present."""
        res, mol_dict = self._create_sample_nerpa_data()
        json_str = format_as_json([res], molecules=mol_dict)
        data = json.loads(json_str)
        assert data[0]["nerpa_score"] == 45.678
        assert data[0]["nerpa_rank"] == 1
        assert data[0]["nerpa_modifications_count"] == 2

    def test_write_structures_with_nerpa_ms_data(self, tmp_path: Path) -> None:
        """Test write_structures includes nerpa_ms_data in exported JSON."""
        nerpa_data = NerpaMSData(
            rank=2,
            score=12.34,
            old_to_new_mon_map=((0, 1),),
            monomer_origins=(),
        )
        monomer = Monomer(
            norine_name="Ala",
            residue="Ala",
            is_methylated=False,
            chirality=Chirality.L,
        )
        dummy_mol = Molecule(
            structure_id="Nerpa_structure",
            monomers={0: monomer},
            bonds=(),
            nerpa_ms_data=nerpa_data,
        )

        write_structures([dummy_mol], output_path=tmp_path)
        expected_file = tmp_path / "structures.json"
        assert expected_file.exists()

        data = json.loads(expected_file.read_text(encoding="utf-8"))
        assert "Nerpa_structure" in data
        assert data["Nerpa_structure"]["nerpa_ms_data"]["rank"] == 2
        assert data["Nerpa_structure"]["nerpa_ms_data"]["score"] == 12.34
