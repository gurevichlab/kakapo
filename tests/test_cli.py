"""Unit tests for the CLI module in cli.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kakapo.cli import main, resolve_paths
from kakapo.config import DEFAULT_OUTPUT_DIR
from kakapo.models import MatchResult, Peak, Spectrum, Tolerance

DUMMY_SPEC = Spectrum(
    spectrum_id="spec_1",
    source_file_path="test_file.mgf",
    peaks=frozenset([Peak(100.0, 1.0)]),
    precursor_mass=500.0,
)


@pytest.fixture
def temp_io_structure(tmp_path: Path):
    """Fixture to set up directory structure with sample files."""
    spec_dir = tmp_path / "spectra"
    spec_dir.mkdir()
    struct_dir = tmp_path / "structures"
    struct_dir.mkdir()

    # Create dummy spectrum file
    spec_file1 = spec_dir / "spec1.mgf"
    spec_file1.write_text("BEGIN IONS\nPEPMASS=100.0\n100.0 50.0\nEND IONS", encoding="utf-8")
    spec_file2 = spec_dir / "spec2.MGF"  # test uppercase extension
    spec_file2.write_text("BEGIN IONS\nPEPMASS=200.0\n200.0 50.0\nEND IONS", encoding="utf-8")

    # Create dummy structure files
    struct_file1 = struct_dir / "struct1.json"
    struct_file1.write_text("{}", encoding="utf-8")
    struct_file2 = struct_dir / "struct2.JSON"  # test uppercase extension
    struct_file2.write_text("{}", encoding="utf-8")

    # Create empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    return {
        "root": tmp_path,
        "spec_dir": spec_dir,
        "struct_dir": struct_dir,
        "empty_dir": empty_dir,
        "spec_file1": spec_file1,
        "spec_file2": spec_file2,
        "struct_file1": struct_file1,
        "struct_file2": struct_file2,
    }


def test_resolve_paths_files(temp_io_structure) -> None:
    """Test resolving a list of specific files."""
    paths = [
        str(temp_io_structure["spec_file1"]),
        str(temp_io_structure["spec_file2"]),
    ]
    resolved = resolve_paths(paths, [".mgf"])
    assert len(resolved) == 2
    assert temp_io_structure["spec_file1"] in resolved
    assert temp_io_structure["spec_file2"] in resolved


def test_resolve_paths_directory(temp_io_structure) -> None:
    """Test resolving directories to retrieve files with matching extensions."""
    resolved_spec = resolve_paths([str(temp_io_structure["spec_dir"])], [".mgf"])
    assert len(resolved_spec) == 2
    assert temp_io_structure["spec_file1"] in resolved_spec
    assert temp_io_structure["spec_file2"] in resolved_spec

    resolved_struct = resolve_paths([str(temp_io_structure["struct_dir"])], [".json"])
    assert len(resolved_struct) == 2
    assert temp_io_structure["struct_file1"] in resolved_struct
    assert temp_io_structure["struct_file2"] in resolved_struct


def test_resolve_paths_json(temp_io_structure) -> None:
    """Test resolving directories with JSON extensions."""
    json_file = temp_io_structure["struct_dir"] / "struct3.json"
    json_file.write_text("{}", encoding="utf-8")

    resolved_struct = resolve_paths(
        [str(temp_io_structure["struct_dir"])],
        [".json"],
    )
    assert len(resolved_struct) == 3
    assert json_file in resolved_struct


def test_resolve_paths_nonexistent() -> None:
    """Test resolving nonexistent paths raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Input path does not exist"):
        resolve_paths(["/nonexistent/file.mgf"], [".mgf"])


def test_resolve_paths_empty_dir(temp_io_structure) -> None:
    """Test resolving directory with no matching files returns empty list."""
    resolved = resolve_paths([str(temp_io_structure["empty_dir"])], [".mgf"])
    assert len(resolved) == 0


class TestCliMainFunction:
    """Test suite for cli.py main function."""

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    def test_main_success(
        self,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test successful run of the pipeline via CLI arguments."""
        mock_run_pipeline.return_value = (
            [
                MatchResult(
                    structure_id="struct1",
                    spectrum_id="spec_1",
                    structure_mass=100.0,
                    spectrum_mass=100.0,
                    tolerance=Tolerance(0.01),
                    matched_peaks_count=2,
                    theoretical_peaks_count=5,
                    experimental_peaks_count=10,
                    monomer_count=1,
                    unmodified_monomer_mass_vector=(100.0,),
                    matched_peaks=[],
                )
            ],
            [],
            [DUMMY_SPEC],
        )

        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.01",
            "--stdout",
        ]

        exit_code = main(args)
        assert exit_code == 0
        mock_run_pipeline.assert_called_once()
        mock_write_results.assert_called_once_with(
            mock_run_pipeline.return_value[0],
            output_path=None,
            format_type="csv",
            molecules=[],
        )
        mock_write_structures.assert_not_called()
        mock_write_spectra.assert_not_called()

    @patch("kakapo.cli.run_pipeline")
    def test_main_num_workers(
        self,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test that --num-workers passes the correct value to run_pipeline."""
        mock_run_pipeline.return_value = ([], [], [])

        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "--stdout",
            "--num-workers",
            "2",
        ]

        exit_code = main(args)
        assert exit_code == 0
        mock_run_pipeline.assert_called_once()
        kwargs = mock_run_pipeline.call_args[1]
        assert kwargs.get("num_workers") == 2

    def test_main_num_workers_invalid(self, temp_io_structure) -> None:
        """Test that invalid --num-workers values raise errors."""
        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "--num-workers",
            "0",
        ]
        with pytest.raises(SystemExit):
            main(args)

        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "--num-workers",
            "-5",
        ]
        with pytest.raises(SystemExit):
            main(args)

    def test_main_missing_arguments(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that missing required arguments raises SystemExit from argparse."""
        with pytest.raises(SystemExit):
            main([])

    @patch("kakapo.cli.resolve_paths")
    def test_main_file_not_found(self, mock_resolve: MagicMock) -> None:
        """Test main returns 1 when paths fail to resolve."""
        mock_resolve.side_effect = FileNotFoundError("File does not exist")
        args = [
            "-s",
            "dummy.mgf",
            "-str",
            "dummy.json",
            "-t",
            "0.5",
        ]
        exit_code = main(args)
        assert exit_code == 1

    @patch("kakapo.cli.run_pipeline")
    def test_main_pipeline_failure(self, mock_run: MagicMock, temp_io_structure) -> None:
        """Test main returns 1 when the pipeline raises an exception."""
        mock_run.side_effect = ValueError("Pipeline failed")
        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
        ]
        exit_code = main(args)
        assert exit_code == 1

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    def test_main_output_file_generation(
        self,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test output file destination path determination based on arguments."""
        mock_run_pipeline.return_value = ([], [], [])
        out_path = temp_io_structure["root"] / "output_test"

        # Case 1: Specific file output path
        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "-o",
            str(out_path / "res.csv"),
            "-f",
            "csv",
        ]
        exit_code = main(args)
        assert exit_code == 0
        mock_write_results.assert_called_with(
            [], output_path=out_path / "res.csv", format_type="csv", molecules=[]
        )
        mock_write_structures.assert_called_with([], output_path=out_path / "res.csv")
        mock_write_spectra.assert_called_with([], output_path=out_path / "res.csv")

        # Case 2: Output directory path (should append filename)
        args_dir = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "-o",
            str(out_path),
            "-f",
            "table",
        ]
        exit_code = main(args_dir)
        assert exit_code == 0
        mock_write_results.assert_called_with(
            [], output_path=out_path / "results.txt", format_type="table", molecules=[]
        )
        mock_write_structures.assert_called_with([], output_path=out_path / "results.txt")
        mock_write_spectra.assert_called_with([], output_path=out_path / "results.txt")

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    def test_main_default_paths(
        self,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test main uses default output and log paths when not provided."""
        mock_run_pipeline.return_value = ([], [], [])

        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
        ]
        exit_code = main(args)
        assert exit_code == 0
        # Should default to DEFAULT_OUTPUT_DIR / results.csv
        expected_out = DEFAULT_OUTPUT_DIR / "results.csv"
        mock_write_results.assert_called_with(
            [], output_path=expected_out, format_type="csv", molecules=[]
        )
        mock_write_structures.assert_called_once_with([], output_path=expected_out)
        mock_write_spectra.assert_called_once_with([], output_path=expected_out)

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    def test_main_dash_stdout(
        self,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test outputting to stdout using '-' output argument."""
        mock_run_pipeline.return_value = ([], [], [])
        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "-o",
            "-",
        ]
        exit_code = main(args)
        assert exit_code == 0
        mock_write_results.assert_called_with(
            [], output_path="-", format_type="csv", molecules=[]
        )
        mock_write_structures.assert_not_called()
        mock_write_spectra.assert_not_called()

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    def test_main_no_resolved_files(
        self,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test main returns 1 when directories exist but no files are matched."""
        args = [
            "-s",
            str(temp_io_structure["empty_dir"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
        ]
        exit_code = main(args)
        assert exit_code == 1

        args_struct = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["empty_dir"]),
            "-t",
            "0.5",
        ]
        exit_code = main(args_struct)
        assert exit_code == 1
        mock_write_results.assert_not_called()

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    @patch("kakapo.cli.setup_logging")
    def test_main_custom_log_file(
        self,
        mock_setup_logging: MagicMock,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test main sets up logging to a custom file path."""
        mock_run_pipeline.return_value = ([], [], [])
        log_file = temp_io_structure["root"] / "custom.log"
        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "-l",
            str(log_file),
        ]
        exit_code = main(args)
        assert exit_code == 0
        mock_setup_logging.assert_called_once_with(verbosity=1, log_file=log_file)
        mock_write_results.assert_called_once_with(
            [], output_path=DEFAULT_OUTPUT_DIR / "results.csv", format_type="csv", molecules=[]
        )
        mock_write_structures.assert_called_once_with(
            [], output_path=DEFAULT_OUTPUT_DIR / "results.csv"
        )
        mock_write_spectra.assert_called_once_with(
            [], output_path=DEFAULT_OUTPUT_DIR / "results.csv"
        )

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    @patch("kakapo.cli.setup_logging")
    def test_main_log_file_in_output_dir_from_directory_path(
        self,
        mock_setup_logging: MagicMock,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test that log file is created in output dir when --output is a directory path."""
        mock_run_pipeline.return_value = ([], [], [])
        out_dir = temp_io_structure["root"] / "specified_output_dir"
        args_dir = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "-o",
            str(out_dir),
        ]
        exit_code = main(args_dir)
        assert exit_code == 0
        mock_setup_logging.assert_called_with(verbosity=1, log_file=out_dir / "kakapo.log")
        mock_write_results.assert_called_once_with(
            [], output_path=out_dir / "results.csv", format_type="csv", molecules=[]
        )
        mock_write_structures.assert_called_once_with([], output_path=out_dir / "results.csv")
        mock_write_spectra.assert_called_once_with([], output_path=out_dir / "results.csv")

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    @patch("kakapo.cli.setup_logging")
    def test_main_log_file_in_output_dir_from_file_path(
        self,
        mock_setup_logging: MagicMock,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test that log file is created in parent dir when --output is a file path."""
        mock_run_pipeline.return_value = ([], [], [])
        out_dir = temp_io_structure["root"] / "specified_output_dir"
        out_file = out_dir / "results.csv"
        args_file = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "-o",
            str(out_file),
        ]
        exit_code = main(args_file)
        assert exit_code == 0
        mock_setup_logging.assert_called_with(verbosity=1, log_file=out_dir / "kakapo.log")
        mock_write_results.assert_called_once_with(
            [], output_path=out_file, format_type="csv", molecules=[]
        )
        mock_write_structures.assert_called_once_with([], output_path=out_file)
        mock_write_spectra.assert_called_once_with([], output_path=out_file)

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    @patch("kakapo.cli.setup_logging")
    def test_main_log_file_in_output_dir_default(
        self,
        mock_setup_logging: MagicMock,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test that log file falls back to DEFAULT_OUTPUT_DIR when --output is not specified."""
        mock_run_pipeline.return_value = ([], [], [])
        args_default = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
        ]
        exit_code = main(args_default)
        assert exit_code == 0
        mock_setup_logging.assert_called_with(
            verbosity=1, log_file=DEFAULT_OUTPUT_DIR / "kakapo.log"
        )
        mock_write_results.assert_called_once_with(
            [], output_path=DEFAULT_OUTPUT_DIR / "results.csv", format_type="csv", molecules=[]
        )
        mock_write_structures.assert_called_once_with(
            [], output_path=DEFAULT_OUTPUT_DIR / "results.csv"
        )
        mock_write_spectra.assert_called_once_with(
            [], output_path=DEFAULT_OUTPUT_DIR / "results.csv"
        )

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    def test_main_top_n_success(
        self,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test successful execution with --top-n slicing the results."""
        results = [
            MatchResult(
                structure_id="struct1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.0,
                tolerance=Tolerance(0.01),
                matched_peaks_count=5,
                theoretical_peaks_count=5,
                experimental_peaks_count=10,
                monomer_count=1,
                unmodified_monomer_mass_vector=(100.0,),
                matched_peaks=[],
            ),
            MatchResult(
                structure_id="struct2",
                spectrum_id="spec_2",
                structure_mass=100.0,
                spectrum_mass=100.0,
                tolerance=Tolerance(0.01),
                matched_peaks_count=3,
                theoretical_peaks_count=5,
                experimental_peaks_count=10,
                monomer_count=1,
                unmodified_monomer_mass_vector=(100.0,),
                matched_peaks=[],
            ),
        ]
        mock_run_pipeline.return_value = (results, [], [DUMMY_SPEC])

        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.01",
            "--top-n",
            "1",
            "--stdout",
        ]

        exit_code = main(args)
        assert exit_code == 0
        mock_run_pipeline.assert_called_once()
        # Should be sliced to top 1 result
        mock_write_results.assert_called_once_with(
            [results[0]],
            output_path=None,
            format_type="csv",
            molecules=[],
        )
        mock_write_structures.assert_not_called()
        mock_write_spectra.assert_not_called()

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    def test_main_multiple_formats(
        self,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test that specifying multiple formats writes one file per format."""
        mock_run_pipeline.return_value = ([], [], [])
        out_path = temp_io_structure["root"] / "multi_out"

        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "-o",
            str(out_path),
            "-f",
            "csv",
            "json",
        ]
        exit_code = main(args)
        assert exit_code == 0
        assert mock_write_results.call_count == 2
        calls = {
            call.kwargs["format_type"]: call.kwargs["output_path"]
            for call in mock_write_results.call_args_list
        }
        assert calls["csv"] == out_path / "results.csv"
        assert calls["json"] == out_path / "results.json"
        mock_write_structures.assert_called_once_with([], output_path=out_path / "results.csv")
        mock_write_spectra.assert_called_once_with([], output_path=out_path / "results.csv")

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.write_structures")
    @patch("kakapo.cli.write_spectra")
    def test_main_write_structures_receives_matched_molecules(
        self,
        mock_write_spectra: MagicMock,
        mock_write_structures: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test that only matched molecules and spectra are filtered and passed to their write methods."""
        from kakapo.models import Chirality, Molecule, Monomer, Peak, Spectrum

        spec2 = Spectrum(
            spectrum_id="spec_2",
            source_file_path="test_file.mgf",
            peaks=frozenset([Peak(100.0, 1.0)]),
            precursor_mass=500.0,
        )

        m1 = Monomer(norine_name="Ala", residue="Ala", is_methylated=False, chirality=Chirality.L)
        mol1 = Molecule(structure_id="struct1", monomers={0: m1})
        mol2 = Molecule(structure_id="struct2", monomers={0: m1})

        results = [
            MatchResult(
                structure_id="struct1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.0,
                tolerance=Tolerance(0.01),
                matched_peaks_count=2,
                theoretical_peaks_count=5,
                experimental_peaks_count=10,
                monomer_count=1,
                unmodified_monomer_mass_vector=(100.0,),
                matched_peaks=[],
            )
        ]

        mock_run_pipeline.return_value = (results, [mol1, mol2], [DUMMY_SPEC, spec2])

        out_path = temp_io_structure["root"] / "output_test"
        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "-o",
            str(out_path),
            "-f",
            "json",
        ]
        exit_code = main(args)
        assert exit_code == 0
        mock_write_structures.assert_called_once_with([mol1], output_path=out_path / "results.json")
        mock_write_spectra.assert_called_once_with(
            [DUMMY_SPEC], output_path=out_path / "results.json"
        )

    def test_main_top_n_invalid(self, temp_io_structure) -> None:
        """Test that invalid values for --top-n raise SystemExit."""
        # Case 1: Zero
        args_zero = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "--top-n",
            "0",
        ]
        with pytest.raises(SystemExit):
            main(args_zero)

        # Case 2: Negative
        args_neg = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "0.5",
            "--top-n",
            "-5",
        ]
        with pytest.raises(SystemExit):
            main(args_neg)

    @patch("kakapo.cli.run_pipeline")
    def test_main_with_ppm_and_min_tolerance(
        self,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test parsing ppm tolerance and min_tolerance CLI options."""
        mock_run_pipeline.return_value = ([], [], [])
        args = [
            "-s",
            str(temp_io_structure["spec_file1"]),
            "-str",
            str(temp_io_structure["struct_file1"]),
            "-t",
            "20ppm",
            "--min-tolerance",
            "0.005",
            "--stdout",
        ]
        exit_code = main(args)
        assert exit_code == 0
        mock_run_pipeline.assert_called_once()
        called_args, called_kwargs = mock_run_pipeline.call_args
        tol = called_kwargs["tolerance"]
        assert tol.value == 20.0
        assert tol.unit == "ppm"
        assert tol.min_tolerance == 0.005

    def test_cli_parser_mode(self) -> None:
        """Verify that --mode / -m CLI flags are parsed correctly."""
        from kakapo.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["-s", "dummy.mgf", "-str", "dummy.json", "--mode", "x-mod"])
        assert args.mode == "x-mod"

        args_short = parser.parse_args(["-s", "dummy.mgf", "-str", "dummy.json", "-m", "two-mod"])
        assert args_short.mode == "two-mod"

        args_default = parser.parse_args(["-s", "dummy.mgf", "-str", "dummy.json"])
        assert args_default.mode == "free-mod"

    def test_cli_parser_scoring(self) -> None:
        """Verify that --scoring CLI flag is parsed correctly."""
        from kakapo.cli import create_parser

        parser = create_parser()

        # Test exp-peaks mode
        args_exp = parser.parse_args(
            ["-s", "dummy.mgf", "-str", "dummy.json", "--scoring", "exp-peaks"]
        )
        assert args_exp.scoring == "exp-peaks"

        # Test intensity mode
        args_intensity = parser.parse_args(
            ["-s", "dummy.mgf", "-str", "dummy.json", "--scoring", "intensity"]
        )
        assert args_intensity.scoring == "intensity"

        # Test default scoring mode (should be exp-peaks)
        args_default = parser.parse_args(["-s", "dummy.mgf", "-str", "dummy.json"])
        assert args_default.scoring == "exp-peaks"

        # Test invalid scoring mode raises SystemExit
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["-s", "dummy.mgf", "-str", "dummy.json", "--scoring", "invalid-mode"]
            )
