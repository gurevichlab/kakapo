"""Unit tests for the MGF parser."""

import json
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
        kakapo_spec = _matchms_to_kakapo(matchms_spec, "test_file.mgf")

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
        assert _matchms_to_kakapo(matchms_spec, "test_file.mgf").charge == 2

    def test_matchms_to_kakapo_with_negative_charge(self):
        """Convert matchms spectrum with negative charge value."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[100.0, 10.0]])
        matchms_spec = MatchmsSpectrum(
            mz=peaks[:, 0],
            intensities=peaks[:, 1],
            metadata={"spectrum_id": "spec_2", "precursor_mz": 400.0, "charge": "1-"},
        )
        assert _matchms_to_kakapo(matchms_spec, "test_file.mgf").charge == -1

    def test_matchms_to_kakapo_with_invalid_charge(self):
        """Convert matchms spectrum with invalid charge, falling back to None."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[100.0, 10.0]])
        matchms_spec = MatchmsSpectrum(
            mz=peaks[:, 0],
            intensities=peaks[:, 1],
            metadata={"spectrum_id": "spec_3", "precursor_mz": 400.0, "charge": "invalid"},
        )
        assert _matchms_to_kakapo(matchms_spec, "test_file.mgf").charge is None

    def test_matchms_to_kakapo_with_pepmass(self):
        """Convert spectrum using pepmass field."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[50.0, 5.0], [100.0, 10.0]])
        metadata = {
            "scan": "42",
            "pepmass": (250.5, 500.0),
        }

        matchms_spec = MatchmsSpectrum(mz=peaks[:, 0], intensities=peaks[:, 1], metadata=metadata)
        kakapo_spec = _matchms_to_kakapo(matchms_spec, "test_file.mgf")

        assert kakapo_spec.spectrum_id == "42"
        assert kakapo_spec.precursor_mass == 250.5

    def test_matchms_to_kakapo_no_peaks(self):
        """Should raise ValueError for spectrum with no peaks."""
        from matchms import Spectrum as MatchmsSpectrum

        metadata = {"spectrum_id": "spec_empty", "precursor_mz": 300.0}

        matchms_spec = MatchmsSpectrum(mz=np.array([]), intensities=np.array([]), metadata=metadata)

        with pytest.raises(ValueError, match="has no peaks"):
            _matchms_to_kakapo(matchms_spec, "test_file.mgf")

    def test_matchms_to_kakapo_invalid_peaks(self):
        """Should raise ValueError for invalid peak data."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[-10.0, 5.0], [100.0, 10.0]])
        metadata = {"spectrum_id": "spec_bad", "precursor_mz": 300.0}

        matchms_spec = MatchmsSpectrum(mz=peaks[:, 0], intensities=peaks[:, 1], metadata=metadata)

        with pytest.raises(ValueError, match="Invalid peak mass"):
            _matchms_to_kakapo(matchms_spec, "test_file.mgf")

    def test_matchms_to_kakapo_invalid_source_file_path_type(self):
        """Should raise TypeError if source_file_path is not a string."""
        from matchms import Spectrum as MatchmsSpectrum

        peaks = np.array([[100.0, 10.0]])
        matchms_spec = MatchmsSpectrum(
            mz=peaks[:, 0],
            intensities=peaks[:, 1],
            metadata={"spectrum_id": "spec_1", "precursor_mz": 400.0},
        )
        with pytest.raises(TypeError, match="Source file path must be a string"):
            _matchms_to_kakapo(matchms_spec, source_file_path=123)  # type: ignore


class TestMgfParserLoadFile:
    """Test loading MGF files."""

    @pytest.fixture(autouse=True)
    def setup_test_cache_dir(self, tmp_path):
        """Setup a temporary cache directory for the test class instance."""
        self.cache_dir = tmp_path / "test_cache"

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
            spectra = load_mgf(mgf_path, self.cache_dir, apply_filters=False)
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
            spectra = load_mgf(mgf_path, self.cache_dir, apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_charge"
            assert spectra[0].precursor_mass == 500.0
            assert spectra[0].charge == 3
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_with_zero_charge(self):
        """Load MGF file containing CHARGE=0+ (treat as unknown/None)."""
        mgf_content = """BEGIN IONS
TITLE=spec_zero_charge
PEPMASS=500.0
CHARGE=0+
100 10
200 20
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            # First load: parses MGF and saves to cache
            spectra = load_mgf(mgf_path, self.cache_dir, apply_filters=False, use_cache=True)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_zero_charge"
            assert spectra[0].charge is None

            # Second load: loads from cache
            spectra_cached = load_mgf(mgf_path, self.cache_dir, apply_filters=False, use_cache=True)
            assert len(spectra_cached) == 1
            assert spectra_cached[0].spectrum_id == "spec_zero_charge"
            assert spectra_cached[0].charge is None
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
            spectra = load_mgf(mgf_path, self.cache_dir, apply_filters=False)
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
            spectra = load_mgf(mgf_path, self.cache_dir, apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].precursor_mass == 550.5
        finally:
            os.unlink(mgf_path)

    def test_load_mgf_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_mgf("/nonexistent/path/file.mgf", self.cache_dir)

    def test_load_mgf_invalid_extension(self):
        """Should raise ValueError for non-.mgf files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="Expected .mgf"):
                load_mgf(txt_path, self.cache_dir)
        finally:
            os.unlink(txt_path)

    def test_load_mgf_empty_file(self):
        """Should raise ValueError for empty MGF file."""
        mgf_content = ""
        mgf_path = self._create_test_mgf(mgf_content)

        try:
            with pytest.raises(ValueError, match="No spectra"):
                load_mgf(mgf_path, self.cache_dir, apply_filters=False)
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
            spectra = load_mgf(mgf_path, self.cache_dir, apply_filters=False, min_peaks=2)
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
            all_spectra = load_mgf_batch(
                [mgf_path_1, mgf_path_2], self.cache_dir, apply_filters=False
            )
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
            spectra = load_mgf_batch(
                [mgf_path_valid, "/nonexistent/file.mgf"], self.cache_dir, apply_filters=False
            )
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
            spectra = load_mgf(Path(mgf_path), self.cache_dir, apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_path"

            # Test with string
            spectra = load_mgf(mgf_path, self.cache_dir, apply_filters=False)
            assert len(spectra) == 1
        finally:
            os.unlink(mgf_path)


class TestMgfParserCaching:
    """Test caching behavior of the MGF parser."""

    def _create_test_mgf(self, content: str, filename: str = "test.mgf") -> str:
        """Helper to create a temporary MGF file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mgf", delete=False, prefix=filename.replace(".mgf", "")
        ) as f:
            f.write(content)
            return f.name

    def test_caching_and_cache_hit(self, tmp_path):
        """Test that loading a file saves it to cache, and reloading hits the cache."""
        mgf_content = """BEGIN IONS
TITLE=spec_cache_test
PEPMASS=500.0
100 10
200 20
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        try:
            # 1. First load: should populate the cache
            spectra_1 = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=True)
            assert len(spectra_1) == 1
            assert spectra_1[0].spectrum_id == "spec_cache_test"

            # Verify cache file exists
            from kakapo.io.mgf_parser import _get_cache_path

            cache_path = _get_cache_path(
                Path(mgf_path),
                apply_filters=False,
                min_peaks=1,
                cache_dir=tmp_path,
            )
            assert cache_path.exists()

            # 2. Modify the cache file content manually to verify cache hit
            with open(cache_path, encoding="utf-8") as f:
                cache_data = json.load(f)

            # Change the ID in cached data
            cache_data[0]["spectrum_id"] = "modified_cache_id"
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)

            # 3. Reload with cache: should load from the modified cache file
            spectra_2 = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=True)
            assert len(spectra_2) == 1
            assert spectra_2[0].spectrum_id == "modified_cache_id"

        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)

    def test_cache_invalidation_on_mtime(self, tmp_path):
        """Test cache invalidation if original file mtime changes."""
        mgf_content = """BEGIN IONS
TITLE=spec_mtime
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        try:
            spectra_1 = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=True)
            assert spectra_1[0].spectrum_id == "spec_mtime"

            from kakapo.io.mgf_parser import _get_cache_path

            cache_path_1 = _get_cache_path(
                Path(mgf_path),
                apply_filters=False,
                min_peaks=1,
                cache_dir=tmp_path,
            )
            assert cache_path_1.exists()

            # Change modification time of original file
            mtime = os.path.getmtime(mgf_path)
            os.utime(mgf_path, (mtime + 10, mtime + 10))

            # Reload: should invalidate the old cache and parse again (different cache path)
            spectra_2 = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=True)
            assert spectra_2[0].spectrum_id == "spec_mtime"

            cache_path_2 = _get_cache_path(
                Path(mgf_path),
                apply_filters=False,
                min_peaks=1,
                cache_dir=tmp_path,
            )
            assert cache_path_2 != cache_path_1
            assert cache_path_2.exists()

        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)

    def test_cache_parameter_isolation(self, tmp_path):
        """Test cache paths differ when loading parameters change."""
        mgf_content = """BEGIN IONS
TITLE=spec_param
PEPMASS=500.0
100 10
200 20
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        try:
            from kakapo.io.mgf_parser import _get_cache_path

            path_obj = Path(mgf_path)

            # Different filters
            cache_1 = _get_cache_path(path_obj, apply_filters=True, min_peaks=1, cache_dir=tmp_path)
            cache_2 = _get_cache_path(
                path_obj, apply_filters=False, min_peaks=1, cache_dir=tmp_path
            )
            assert cache_1 != cache_2

            # Different min_peaks
            cache_3 = _get_cache_path(path_obj, apply_filters=True, min_peaks=1, cache_dir=tmp_path)
            cache_4 = _get_cache_path(path_obj, apply_filters=True, min_peaks=3, cache_dir=tmp_path)
            assert cache_3 != cache_4

        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)

    def test_cache_corrupted_fallback(self, tmp_path):
        """Test graceful fallback to raw parsing if cache JSON is corrupted."""
        mgf_content = """BEGIN IONS
TITLE=spec_corrupt
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        try:
            load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=True)

            from kakapo.io.mgf_parser import _get_cache_path

            cache_path = _get_cache_path(
                Path(mgf_path),
                apply_filters=False,
                min_peaks=1,
                cache_dir=tmp_path,
            )
            assert cache_path.exists()

            # Corrupt the cache file with invalid JSON
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write("{invalid json...")

            # Load should succeed by falling back to raw MGF parsing
            spectra = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=True)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_corrupt"

        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)

    def test_use_cache_false_flag(self, tmp_path):
        """Test that use_cache=False does not create or load cache files."""
        mgf_content = """BEGIN IONS
TITLE=spec_no_cache
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        try:
            spectra = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=False)
            assert len(spectra) == 1

            from kakapo.io.mgf_parser import _get_cache_path

            cache_path = _get_cache_path(
                Path(mgf_path),
                apply_filters=False,
                min_peaks=1,
                cache_dir=tmp_path,
            )
            assert not cache_path.exists()

        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)

    def test_explicit_cache_dir_parameter(self, tmp_path):
        """Test passing an explicit cache_dir to load_mgf overrides DEFAULT_CACHE_DIR."""
        mgf_content = """BEGIN IONS
TITLE=spec_explicit_dir
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        custom_dir = tmp_path / "custom_cache_location"

        try:
            # Load with explicit cache_dir
            spectra = load_mgf(
                mgf_path,
                custom_dir,
                apply_filters=False,
                use_cache=True,
            )
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_explicit_dir"

            # Verify cache was created in custom_dir and not in DEFAULT_CACHE_DIR
            from kakapo.io.mgf_parser import _get_cache_path

            cache_path = _get_cache_path(
                Path(mgf_path),
                apply_filters=False,
                min_peaks=1,
                cache_dir=custom_dir,
            )
            assert cache_path.exists()
            assert custom_dir in cache_path.parents

        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)

    def test_source_file_path_propagation(self, tmp_path):
        """Test that the source_file_path is populated on parsed spectra."""
        mgf_content = """BEGIN IONS
TITLE=spec_source_path_test
PEPMASS=500.0
100 10
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        try:
            # 1. Test raw loading
            spectra = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=False)
            assert len(spectra) == 1
            assert spectra[0].source_file_path == str(Path(mgf_path).resolve())

            # 2. Test loading from cache
            spectra_cached = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=True)
            assert len(spectra_cached) == 1
            assert spectra_cached[0].source_file_path == str(Path(mgf_path).resolve())

        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)
