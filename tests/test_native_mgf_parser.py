"""Unit tests for the native MGF parser module (kakapo.io.native_mgf_parser)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from kakapo.io.native_mgf_parser import (
    _parse_mgf_block,
    _parse_mgf_chunk_worker,
    apply_spectrum_filters,
    parse_charge,
    parse_mgf_file,
    parse_precursor_mass,
    parse_spectrum_id,
)
from kakapo.models import Spectrum


class TestParseCharge:
    """Test parse_charge function."""

    def test_parse_charge_none(self):
        """Test parse_charge with None input."""
        assert parse_charge(None) is None

    def test_parse_charge_numeric(self):
        """Test parse_charge with integer and float inputs."""
        assert parse_charge(2) == 2
        assert parse_charge(-3) == -3
        assert parse_charge(2.0) == 2
        assert parse_charge(-3.0) == -3
        assert parse_charge(0) is None
        assert parse_charge(0.0) is None

    def test_parse_charge_invalid_types(self):
        """Test parse_charge with unparseable non-string/numeric types."""
        assert parse_charge([1, 2]) is None
        assert parse_charge({"charge": 2}) is None
        assert parse_charge(object()) is None

    def test_parse_charge_string_formats(self):
        """Test parse_charge with various string representations."""
        assert parse_charge("") is None
        assert parse_charge("   ") is None
        assert parse_charge("3+") == 3
        assert parse_charge("2-") == -2
        assert parse_charge("+2") == 2
        assert parse_charge("-2") == -2
        assert parse_charge("+2+") == 2
        assert parse_charge("-3-") == -3
        assert parse_charge(" 4 + ") == 4
        assert parse_charge("2.0") == 2
        assert parse_charge("0") is None
        assert parse_charge("0+") is None
        assert parse_charge("0-") is None
        assert parse_charge("invalid") is None


class TestParsePrecursorMass:
    """Test parse_precursor_mass function."""

    def test_parse_precursor_mass_none(self):
        """Test parse_precursor_mass with None input."""
        assert parse_precursor_mass(None) is None

    def test_parse_precursor_mass_numeric(self):
        """Test parse_precursor_mass with integer and float inputs."""
        assert parse_precursor_mass(500.5) == 500.5
        assert parse_precursor_mass(0) == 0.0
        assert parse_precursor_mass(-10.0) is None

    def test_parse_precursor_mass_invalid_types(self):
        """Test parse_precursor_mass with unparseable types."""
        assert parse_precursor_mass([500.5]) is None
        assert parse_precursor_mass({"mass": 500.5}) is None

    def test_parse_precursor_mass_string_formats(self):
        """Test parse_precursor_mass with various string inputs."""
        assert parse_precursor_mass("") is None
        assert parse_precursor_mass("   ") is None
        assert parse_precursor_mass("500.5") == 500.5
        assert parse_precursor_mass("500.5 1000.0") == 500.5
        assert parse_precursor_mass("-100.0") is None
        assert parse_precursor_mass("invalid") is None


class TestParseSpectrumId:
    """Test parse_spectrum_id function."""

    def test_parse_spectrum_id_title_priority(self):
        """Test that TITLE takes highest priority over other ID keys."""
        meta = {
            "TITLE": "spec_title_1",
            "SPECTRUM_ID": "spec_id_1",
            "FEATURE_ID": "feat_1",
        }
        assert parse_spectrum_id(meta, 1) == "spec_title_1"

    def test_parse_spectrum_id_candidate_keys(self):
        """Test metadata fallback ID keys in order when TITLE is absent or empty."""
        assert parse_spectrum_id({"SPECTRUM_ID": "id_1"}, 1) == "id_1"
        assert parse_spectrum_id({"ID": "id_2"}, 1) == "id_2"
        assert parse_spectrum_id({"FEATURE_ID": "id_3"}, 1) == "id_3"
        assert parse_spectrum_id({"FEATURE": "id_4"}, 1) == "id_4"
        assert parse_spectrum_id({"SCANS": "id_5"}, 1) == "id_5"
        assert parse_spectrum_id({"SCAN": "id_6"}, 1) == "id_6"

    def test_parse_spectrum_id_fallback(self):
        """Test fallback spectrum ID generation when no metadata ID keys match."""
        assert parse_spectrum_id({}, 42) == "spectrum_42"
        assert parse_spectrum_id({"TITLE": "   "}, 5) == "spectrum_5"


class TestParseMgfBlock:
    """Test _parse_mgf_block helper function."""

    def test_parse_mgf_block_valid(self):
        """Test parsing a valid MGF block body."""
        body = """TITLE=test_spec
PEPMASS=500.0
CHARGE=2+
# Comment line
; Another comment
100.0 10.0
200.0, 20.0
"""
        spec = _parse_mgf_block(body, "test.mgf", 1)
        assert spec is not None
        assert spec.spectrum_id == "test_spec"
        assert spec.precursor_mass == 500.0
        assert spec.charge == 2
        assert len(spec.mzs) == 2
        assert len(spec.intensities) == 2

    def test_parse_mgf_block_missing_precursor(self):
        """Test parsing block with missing precursor mass returns None."""
        body = """TITLE=no_pepmass
100.0 10.0
"""
        assert _parse_mgf_block(body, "test.mgf", 1) is None

    def test_parse_mgf_block_no_peaks(self):
        """Test parsing block with no peak data returns None."""
        body = """TITLE=no_peaks
PEPMASS=500.0
"""
        assert _parse_mgf_block(body, "test.mgf", 1) is None

    def test_parse_mgf_block_invalid_peaks_filtered(self):
        """Test that invalid or negative peak values are ignored."""
        body = """TITLE=filtered_peaks
PEPMASS=500.0
-10.0 10.0
100.0 -5.0
abc def
200.0 20.0
"""
        spec = _parse_mgf_block(body, "test.mgf", 1)
        assert spec is not None
        assert len(spec.mzs) == 1
        assert spec.mzs[0] == 200.0

    def test_parse_mgf_block_spectrum_exception(self):
        """Test graceful failure when Spectrum creation raises an exception."""
        body = """TITLE=err_spec
PEPMASS=500.0
100.0 10.0
"""
        with patch("kakapo.io.native_mgf_parser.Spectrum", side_effect=ValueError("Test error")):
            assert _parse_mgf_block(body, "test.mgf", 1) is None


class TestApplySpectrumFilters:
    """Test apply_spectrum_filters function."""

    def test_apply_spectrum_filters_none_input(self):
        """Test apply_spectrum_filters with None input."""
        assert apply_spectrum_filters(None, []) is None

    def test_apply_spectrum_filters_pipeline(self):
        """Test apply_spectrum_filters executes filter pipeline correctly."""

        def filter_add_tag(spec: Spectrum) -> Spectrum:
            return spec

        def filter_with_kwarg(spec: Spectrum, factor: float = 1.0) -> Spectrum:
            return spec

        def filter_reject(spec: Spectrum) -> None:
            return None

        spec = Spectrum(
            spectrum_id="spec_1",
            source_file_path="test.mgf",
            mzs=np.array([100.0], dtype=np.float64),
            intensities=np.array([10.0], dtype=np.float64),
            precursor_mass=500.0,
            charge=2,
        )

        filters = [filter_add_tag, (filter_with_kwarg, {"factor": 2.0})]
        res = apply_spectrum_filters(spec, filters)
        assert res is not None

        filters_with_rejection = [filter_add_tag, filter_reject, filter_with_kwarg]
        assert apply_spectrum_filters(spec, filters_with_rejection) is None


class TestParseMgfChunkWorker:
    """Test _parse_mgf_chunk_worker function."""

    def test_worker_parsing(self):
        """Test worker parsing of multiple block bodies."""
        b1 = "TITLE=spec1\nPEPMASS=500.0\n100 10\n200 20"
        b2 = "TITLE=spec2\nPEPMASS=600.0\n150 15"

        args = ([b1, b2], "test.mgf", 0, False, 1)
        spectra = _parse_mgf_chunk_worker(args)
        assert len(spectra) == 2
        assert spectra[0].spectrum_id == "spec1"
        assert spectra[1].spectrum_id == "spec2"

    def test_worker_min_peaks_filtering(self):
        """Test worker applies min_peaks filtering correctly."""
        b1 = "TITLE=spec1\nPEPMASS=500.0\n100 10"
        b2 = "TITLE=spec2\nPEPMASS=600.0\n100 10\n200 20\n300 30"

        args = ([b1, b2], "test.mgf", 0, False, 2)
        spectra = _parse_mgf_chunk_worker(args)
        assert len(spectra) == 1
        assert spectra[0].spectrum_id == "spec2"


class TestParseMgfFile:
    """Test parse_mgf_file function."""

    def _create_test_mgf(self, content: str, filename: str = "test.mgf") -> str:
        """Helper to create a temporary MGF file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mgf", delete=False, prefix=filename.replace(".mgf", "")
        ) as f:
            f.write(content)
            return f.name

    def test_parse_mgf_file_nonexistent(self):
        """Test FileNotFoundError for non-existent file path."""
        with pytest.raises(FileNotFoundError, match="MGF file not found"):
            parse_mgf_file("/nonexistent/path/file.mgf")

    def test_parse_mgf_file_no_begin_ions(self):
        """Test parse_mgf_file with file lacking BEGIN IONS tags."""
        mgf_path = self._create_test_mgf("SOME OTHER CONTENT\nNO IONS HERE")
        try:
            spectra = parse_mgf_file(mgf_path)
            assert spectra == []
        finally:
            os.unlink(mgf_path)

    def test_parse_mgf_file_empty_blocks(self):
        """Test parse_mgf_file with incomplete blocks."""
        mgf_path = self._create_test_mgf("BEGIN IONS\nNO END IONS HERE")
        try:
            spectra = parse_mgf_file(mgf_path)
            assert spectra == []
        finally:
            os.unlink(mgf_path)

    def test_parse_mgf_file_single_worker(self):
        """Test parse_mgf_file in single-worker mode."""
        content = """BEGIN IONS
TITLE=spec_1
PEPMASS=500.0
100 10
200 20
END IONS
BEGIN IONS
TITLE=spec_2
PEPMASS=600.0
150 15
250 25
END IONS
"""
        mgf_path = self._create_test_mgf(content)
        try:
            spectra = parse_mgf_file(mgf_path, num_workers=1)
            assert len(spectra) == 2
            assert spectra[0].spectrum_id == "spec_1"
            assert spectra[1].spectrum_id == "spec_2"
        finally:
            os.unlink(mgf_path)

    def test_parse_mgf_file_parallel_workers(self):
        """Test parse_mgf_file with parallel process pool execution."""
        blocks = []
        for i in range(60):
            blocks.append(f"BEGIN IONS\nTITLE=spec_{i}\nPEPMASS=500.0\n100 10\n200 20\nEND IONS")
        content = "\n".join(blocks)
        mgf_path = self._create_test_mgf(content)
        try:
            spectra = parse_mgf_file(mgf_path, num_workers=2)
            assert len(spectra) == 60
            assert spectra[0].spectrum_id == "spec_0"
            assert spectra[59].spectrum_id == "spec_59"
        finally:
            os.unlink(mgf_path)

    def test_parse_mgf_file_source_path_override(self):
        """Test passing custom source_file_path to parse_mgf_file."""
        content = """BEGIN IONS
TITLE=spec_1
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path = self._create_test_mgf(content)
        try:
            custom_source = "custom/path/sample.mgf"
            spectra = parse_mgf_file(Path(mgf_path), source_file_path=custom_source)
            assert len(spectra) == 1
            assert spectra[0].source_file_path == custom_source
        finally:
            os.unlink(mgf_path)

    def test_parse_mgf_file_read_error(self):
        """Test ValueError when file read fails."""
        mgf_path = self._create_test_mgf("BEGIN IONS\nEND IONS")
        try:
            with patch("builtins.open", side_effect=PermissionError("Permission denied")):
                with pytest.raises(ValueError, match="Error reading MGF file"):
                    parse_mgf_file(mgf_path)
        finally:
            os.unlink(mgf_path)

    def test_parse_mgf_file_parallel_executor_error(self):
        """Test ValueError when ProcessPoolExecutor raises an exception."""
        blocks = []
        for i in range(60):
            blocks.append(f"BEGIN IONS\nTITLE=spec_{i}\nPEPMASS=500.0\n100 10\nEND IONS")
        content = "\n".join(blocks)
        mgf_path = self._create_test_mgf(content)
        try:
            with patch(
                "kakapo.io.native_mgf_parser.ProcessPoolExecutor",
                side_effect=RuntimeError("Process pool failed"),
            ):
                with pytest.raises(ValueError, match="Parallel MGF parsing failed"):
                    parse_mgf_file(mgf_path, num_workers=2)
        finally:
            os.unlink(mgf_path)
