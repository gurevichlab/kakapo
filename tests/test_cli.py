"""Unit tests for the CLI module in cli.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kakapo.cli import main, resolve_paths
from kakapo.config import DEFAULT_OUTPUT_DIR
from kakapo.models import MatchResult


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
    struct_file1 = struct_dir / "struct1.yaml"
    struct_file1.write_text("compound_id: struct1\nmonomers:\n  0: {name: Ala, atoms: [0]}\natomic_bonds: []", encoding="utf-8")
    struct_file2 = struct_dir / "struct2.YML"  # test uppercase extension
    struct_file2.write_text("compound_id: struct2\nmonomers:\n  0: {name: Gly, atoms: [0]}\natomic_bonds: []", encoding="utf-8")

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

    resolved_struct = resolve_paths([str(temp_io_structure["struct_dir"])], [".yaml", ".yml"])
    assert len(resolved_struct) == 2
    assert temp_io_structure["struct_file1"] in resolved_struct
    assert temp_io_structure["struct_file2"] in resolved_struct


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
    def test_main_success(
        self,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test successful run of the pipeline via CLI arguments."""
        mock_run_pipeline.return_value = [
            MatchResult(
                structure_id="struct1",
                spectrum_id="spec_1",
                structure_mass=100.0,
                spectrum_mass=100.0,
                mass_tolerance=0.01,
                matched_peaks_count=2,
                theoretical_peaks_count=5,
                experimental_peaks_count=10,
                monomer_count=1,
                matched_peaks=[],
            )
        ]

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
            mock_run_pipeline.return_value,
            output_path=None,
            format_type="table",
        )

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
            "dummy.yaml",
            "-t",
            "0.5",
        ]
        exit_code = main(args)
        assert exit_code == 1

    @patch("kakapo.cli.run_pipeline")
    def test_main_pipeline_failure(
        self, mock_run: MagicMock, temp_io_structure
    ) -> None:
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
    def test_main_output_file_generation(
        self,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test output file destination path determination based on arguments."""
        mock_run_pipeline.return_value = []
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
        mock_write_results.assert_called_with([], output_path=out_path / "res.csv", format_type="csv")

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
        mock_write_results.assert_called_with([], output_path=out_path / "results.txt", format_type="table")

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    def test_main_default_paths(
        self,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test main uses default output and log paths when not provided."""
        mock_run_pipeline.return_value = []

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
        # Should default to DEFAULT_OUTPUT_DIR / results.txt
        expected_out = DEFAULT_OUTPUT_DIR / "results.txt"
        mock_write_results.assert_called_with([], output_path=expected_out, format_type="table")

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    def test_main_dash_stdout(
        self,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test outputting to stdout using '-' output argument."""
        mock_run_pipeline.return_value = []
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
        mock_write_results.assert_called_with([], output_path="-", format_type="table")

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

    @patch("kakapo.cli.run_pipeline")
    @patch("kakapo.cli.write_results")
    @patch("kakapo.cli.setup_logging")
    def test_main_custom_log_file(
        self,
        mock_setup_logging: MagicMock,
        mock_write_results: MagicMock,
        mock_run_pipeline: MagicMock,
        temp_io_structure,
    ) -> None:
        """Test main sets up logging to a custom file path."""
        mock_run_pipeline.return_value = []
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
        mock_setup_logging.assert_called_once_with(verbosity=0, log_file=log_file)

