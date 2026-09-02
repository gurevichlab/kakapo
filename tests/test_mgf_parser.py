"""Unit tests for the MGF parser."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from kakapo.io.mgf_parser import (
    load_mgf,
    load_mgf_batch,
)


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
            assert len(spectra[0].mzs) == 3
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
            spectra = load_mgf(mgf_path, self.cache_dir, apply_filters=False, use_cache=True)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_zero_charge"
            assert spectra[0].charge is None

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
            spectra = load_mgf(Path(mgf_path), self.cache_dir, apply_filters=False)
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_path"

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
            spectra_1 = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=True)
            assert len(spectra_1) == 1
            assert spectra_1[0].spectrum_id == "spec_cache_test"

            from kakapo.io.mgf_parser import _get_cache_path

            cache_path = _get_cache_path(
                Path(mgf_path),
                apply_filters=False,
                min_peaks=1,
                cache_dir=tmp_path,
            )
            assert cache_path.exists()

            with open(cache_path, encoding="utf-8") as f:
                cache_data = json.load(f)

            cache_data[0]["spectrum_id"] = "modified_cache_id"
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f)

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

            mtime = os.path.getmtime(mgf_path)
            os.utime(mgf_path, (mtime + 10, mtime + 10))

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

            cache_1 = _get_cache_path(path_obj, apply_filters=True, min_peaks=1, cache_dir=tmp_path)
            cache_2 = _get_cache_path(
                path_obj, apply_filters=False, min_peaks=1, cache_dir=tmp_path
            )
            assert cache_1 != cache_2

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

            with open(cache_path, "w", encoding="utf-8") as f:
                f.write("{invalid json...")

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
            spectra = load_mgf(
                mgf_path,
                custom_dir,
                apply_filters=False,
                use_cache=True,
            )
            assert len(spectra) == 1
            assert spectra[0].spectrum_id == "spec_explicit_dir"

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
            spectra = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=False)
            assert len(spectra) == 1
            assert spectra[0].source_file_path == str(Path(mgf_path).resolve())

            spectra_cached = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=True)
            assert len(spectra_cached) == 1
            assert spectra_cached[0].source_file_path == str(Path(mgf_path).resolve())

        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)

    def test_load_mgf_with_duplicate_ids(self, tmp_path):
        """Test that duplicate spectrum IDs get suffix _idx appended."""
        mgf_content = """BEGIN IONS
TITLE=spec_dup
PEPMASS=500.0
100 10
END IONS
BEGIN IONS
TITLE=spec_dup
PEPMASS=600.0
200 20
END IONS
BEGIN IONS
TITLE=spec_unique
PEPMASS=700.0
300 30
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        try:
            spectra = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=False)
            assert len(spectra) == 3
            assert spectra[0].spectrum_id == "spec_dup_0"
            assert spectra[1].spectrum_id == "spec_dup_1"
            assert spectra[2].spectrum_id == "spec_unique"
        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)

    def test_load_mgf_with_missing_ids(self, tmp_path):
        """Test that spectra with no ID fields generate ID as index."""
        mgf_content = """BEGIN IONS
PEPMASS=500.0
100 10
END IONS
BEGIN IONS
PEPMASS=600.0
200 20
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        try:
            spectra = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=False)
            assert len(spectra) == 2
            assert spectra[0].spectrum_id == "spectrum_1"
            assert spectra[1].spectrum_id == "spectrum_2"
        finally:
            os.unlink(mgf_path)

    def test_parse_spectrum_id_candidates(self, tmp_path):
        """Test spectrum ID candidate extraction with TITLE taking first priority."""
        mgf_content = """BEGIN IONS
TITLE=spec_title_1
FEATURE_ID=feat_123
PEPMASS=500.0
100 10
END IONS
BEGIN IONS
FEATURE=feat_456
PEPMASS=600.0
200 20
END IONS
BEGIN IONS
SPECTRUM_ID=spec_789
PEPMASS=700.0
300 30
END IONS
"""
        mgf_path = self._create_test_mgf(mgf_content)
        try:
            spectra = load_mgf(mgf_path, tmp_path, apply_filters=False, use_cache=False)
            assert len(spectra) == 3
            assert spectra[0].spectrum_id == "spec_title_1"
            assert spectra[1].spectrum_id == "feat_456"
            assert spectra[2].spectrum_id == "spec_789"
        finally:
            if os.path.exists(mgf_path):
                os.unlink(mgf_path)

