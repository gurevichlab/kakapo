"""Unit tests for the MGF parser."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from kakapo.io.mgf_parser import (
    _get_precursor_mass,
    _get_spectrum_id,
    _matchms_to_kakapo,
    load_mgf,
    load_mgf_batch,
)
from kakapo.models import Spectrum


class TestMgfParserHelpers:
    """Test helper functions for MGF parsing."""

    def test_get_precursor_mass_from_precursor_mz(self):
        """Extract precursor mass from precursor_mz field."""
        metadata = {
            "precursor_mz": 500.5,
            "spectrum_id": "spec_1",
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 500.5

    def test_get_precursor_mass_from_pepmass_float(self):
        """Extract precursor mass from pepmass as float."""
        metadata = {
            "pepmass": 600.0,
            "spectrum_id": "spec_2",
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 600.0

    def test_get_precursor_mass_from_pepmass_tuple(self):
        """Extract precursor mass from pepmass as tuple [mass, intensity]."""
        metadata = {
            "pepmass": (700.5, 1000.0),
            "spectrum_id": "spec_3",
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 700.5

    def test_get_precursor_mass_from_pepmass_list(self):
        """Extract precursor mass from pepmass as list."""
        metadata = {
            "pepmass": [750.0, 500.0],
            "spectrum_id": "spec_4",
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 750.0

    def test_get_precursor_mass_priority_precursor_mz(self):
        """precursor_mz should be prioritized over pepmass."""
        metadata = {
            "precursor_mz": 400.0,
            "pepmass": 500.0,
        }
        mass = _get_precursor_mass(metadata)
        assert mass == 400.0

    def test_get_precursor_mass_missing_field(self):
        """Should raise ValueError when no precursor mass field is present."""
        metadata = {
            "spectrum_id": "spec_5",
        }
        with pytest.raises(ValueError, match="Could not extract"):
            _get_precursor_mass(metadata)

    def test_get_precursor_mass_negative_value(self):
        """Should raise ValueError for negative precursor mass."""
        metadata = {
            "precursor_mz": -100.0,
        }
        with pytest.raises(ValueError, match="Could not extract"):
            _get_precursor_mass(metadata)

    def test_get_spectrum_id_from_spectrum_id_field(self):
        """Extract spectrum_id from spectrum_id field."""
        metadata = {"spectrum_id": "spec_001"}
        spectrum_id = _get_spectrum_id(metadata)
        assert spectrum_id == "spec_001"

    def test_get_spectrum_id_from_scan_field(self):
        """Extract spectrum_id from scan field."""
        metadata = {"scan": "123"}
        spectrum_id = _get_spectrum_id(metadata)
        assert spectrum_id == "123"

    def test_get_spectrum_id_from_title_field(self):
        """Extract spectrum_id from title field."""
        metadata = {"title": "compound_A_ms2"}
        spectrum_id = _get_spectrum_id(metadata)
        assert spectrum_id == "compound_A_ms2"

    def test_get_spectrum_id_fallback(self):
        """Fall back to 'unknown' when no ID field is present."""
        metadata = {"precursor_mz": 500.0}
        spectrum_id = _get_spectrum_id(metadata)
        assert spectrum_id == "unknown"


class TestMgfParserConversion:
    """Test conversion from matchms to kakapo Spectrum."""

    def test_matchms_to_kakapo_basic_conversion(self):
        """Convert a basic matchms spectrum to kakapo Spectrum."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[100.0, 10.0], [200.0, 20.0], [300.0, 15.0]])
        metadata = {
            "spectrum_id": "spec_001",
            "precursor_mz": 400.0,
        }

        matchms_spec = MatchmsSpectrum(mz=peaks[:, 0], intensities=peaks[:, 1], metadata=metadata)
        kakapo_spec = _matchms_to_kakapo(matchms_spec)

        assert isinstance(kakapo_spec, Spectrum)
        assert kakapo_spec.spectrum_id == "spec_001"
        assert kakapo_spec.precursor_mass == 400.0
        assert len(kakapo_spec.peaks) == 3
        assert kakapo_spec.charge is None

    def test_matchms_to_kakapo_with_positive_charge(self):
        """Convert matchms spectrum with positive charge value."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[100.0, 10.0]])
        matchms_spec = MatchmsSpectrum(
            mz=peaks[:, 0],
            intensities=peaks[:, 1],
            metadata={"spectrum_id": "spec_1", "precursor_mz": 400.0, "charge": "2+"},
        )
        assert _matchms_to_kakapo(matchms_spec).charge == 2

    def test_matchms_to_kakapo_with_negative_charge(self):
        """Convert matchms spectrum with negative charge value."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[100.0, 10.0]])
        matchms_spec = MatchmsSpectrum(
            mz=peaks[:, 0],
            intensities=peaks[:, 1],
            metadata={"spectrum_id": "spec_2", "precursor_mz": 400.0, "charge": "1-"},
        )
        assert _matchms_to_kakapo(matchms_spec).charge == -1

    def test_matchms_to_kakapo_with_invalid_charge(self):
        """Convert matchms spectrum with invalid charge, falling back to None."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[100.0, 10.0]])
        matchms_spec = MatchmsSpectrum(
            mz=peaks[:, 0],
            intensities=peaks[:, 1],
            metadata={"spectrum_id": "spec_3", "precursor_mz": 400.0, "charge": "invalid"},
        )
        assert _matchms_to_kakapo(matchms_spec).charge is None

    def test_matchms_to_kakapo_with_pepmass(self):
        """Convert spectrum using pepmass field."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[50.0, 5.0], [100.0, 10.0]])
        metadata = {
            "scan": "42",
            "pepmass": (250.5, 500.0),
        }

        matchms_spec = MatchmsSpectrum(mz=peaks[:, 0], intensities=peaks[:, 1], metadata=metadata)
        kakapo_spec = _matchms_to_kakapo(matchms_spec)

        assert kakapo_spec.spectrum_id == "42"
        assert kakapo_spec.precursor_mass == 250.5

    def test_matchms_to_kakapo_no_peaks(self):
        """Should raise ValueError for spectrum with no peaks."""
        from matchms import Spectrum as MatchmsSpectrum

        metadata = {"spectrum_id": "spec_empty", "precursor_mz": 300.0}

        matchms_spec = MatchmsSpectrum(mz=np.array([]), intensities=np.array([]), metadata=metadata)

        with pytest.raises(ValueError, match="has no peaks"):
            _matchms_to_kakapo(matchms_spec)

    def test_matchms_to_kakapo_invalid_peaks(self):
        """Should raise ValueError for invalid peak data."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[-10.0, 5.0], [100.0, 10.0]])
        metadata = {"spectrum_id": "spec_bad", "precursor_mz": 300.0}

        matchms_spec = MatchmsSpectrum(mz=peaks[:, 0], intensities=peaks[:, 1], metadata=metadata)

        with pytest.raises(ValueError, match="Invalid peak mass"):
            _matchms_to_kakapo(matchms_spec)


class TestMgfParserLoadFile:
    """Test loading MGF files."""

    def _create_test_mgf(self, content: str, filename: str = "test.mgf") -> str:
        """Helper to create a temporary MGF file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mgf", delete=False, prefix=filename.replace(".mgf", "")
        ) as f:
            f.write(content)
            return f.name

    def test_load_mgf_basic_file(self):
        """Load a basic MGF file with one spectrum."""
        mgf_content = """BEGIN IONS
TITLE=spectrum_1
PEPMASS=500.0
100 10
200 20
300 15
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            spectra = load_mgf(mgf_path, apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spectrum_1"
            assert spectra[0].precursor_mass == 500.0
            assert len(spectra[0].peaks) == 3
            assert spectra[0].charge is None
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_with_charge(self):
        """Load MGF file containing charge field."""
        mgf_content = """BEGIN IONS
TITLE=spec_charge
PEPMASS=500.0
CHARGE=3+
100 10
200 20
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            spectra = load_mgf(mgf_path, apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_charge"
            assert spectra[0].precursor_mass == 500.0
            assert spectra[0].charge == 3
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_multiple_spectra(self):
        """Load MGF file with multiple spectra."""
        mgf_content = """BEGIN IONS
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
350 35
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            spectra = load_mgf(mgf_path, apply_filters=False)
            assert len(spectra) == 2
            assert spectra[0].spectrum_id == "spec_1"
            assert spectra[1].spectrum_id == "spec_2"
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_with_precursor_mz_field(self):
        """Load MGF using PRECURSORMZ field (common format variant)."""
        mgf_content = """BEGIN IONS
TITLE=spec_mz
PRECURSORMZ=550.5
100 10
200 20
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            spectra = load_mgf(mgf_path, apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].precursor_mass == 550.5
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_mgf("/nonexistent/path/file.mgf")

    def test_load_mgf_invalid_extension(self):
        """Should raise ValueError for non-.mgf files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="Expected .mgf"):
                load_mgf(txt_path)
        finally:
            os.unlink(txt_path)

    def test_load_mgf_empty_file(self):
        """Should raise ValueError for empty MGF file."""
        mgf_content = ""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            with pytest.raises(ValueError, match="No spectra"):
                load_mgf(mgf_path, apply_filters=False)
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_min_peaks_filter(self):
        """Should filter out spectra with too few peaks."""
        mgf_content = """BEGIN IONS
TITLE=few_peaks
PEPMASS=500.0
100 10
END IONS
BEGIN IONS
TITLE=many_peaks
PEPMASS=600.0
100 10
200 20
300 30
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            spectra = load_mgf(mgf_path, apply_filters=False, min_peaks=2)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "many_peaks"
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_batch_multiple_files(self):
        """Load multiple MGF files in batch."""
        mgf_content_1 = """BEGIN IONS
TITLE=spec_file1
PEPMASS=500.0
100 10
200 20
END IONS
"""
        mgf_content_2 = """BEGIN IONS
TITLE=spec_file2
PEPMASS=600.0
150 15
250 25
END IONS
"""
        mgf_path_1 = self._create_test_mgf(mgf_content_1, "file1.mgf")
        mgf_path_2 = self._create_test_mgf(mgf_content_2, "file2.mgf")

        try:
            all_spectra = load_mgf_batch([mgf_path_1, mgf_path_2], apply_filters=False)
            assert len(all_spectra) == 2
            assert all_spectra[0].spectrum_id == "spec_file1"
            assert all_spectra[1].spectrum_id == "spec_file2"
        finally:
            os.unlink(mgf_path_1)
            os.unlink(mgf_path_2)

    def test_load_mgf_batch_with_error(self):
        """Batch loading should continue and load valid files even if one file fails."""
        mgf_content_valid = """BEGIN IONS
TITLE=valid
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path_valid = self._create_test_mgf(mgf_content_valid, "valid.mgf")

        try:
            spectra = load_mgf_batch([mgf_path_valid, "/nonexistent/file.mgf"], apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "valid"
        finally:
            os.unlink(mgf_path_valid)

    def test_load_mgf_with_paths_object(self):
        """load_mgf should accept both str and Path objects."""
        mgf_content = """BEGIN IONS
TITLE=spec_path
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            # Test with Path object
            spectra = load_mgf(Path(mgf_path), apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_path"

            # Test with string
            spectra = load_mgf(mgf_path, apply_filters=False)
            assert len(spectra) == 1
        finally:
            os.unlink(mgf_path)
