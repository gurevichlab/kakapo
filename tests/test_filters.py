"""Tests for custom native kakapo filters."""

import numpy as np
import pytest

from kakapo.models import Spectrum
from kakapo.utils.filters import (
    keep_highest_intensity_in_radius,
    reduce_to_number_of_peaks,
    remove_peaks_relative_to_precursor_mz_charge_aware,
    require_minimum_number_of_high_peaks,
    select_by_relative_intensity,
)


def make_spectrum(mzs, intensities, precursor_mass=500.0, charge=1, spec_id="spec1"):
    return Spectrum(
        spectrum_id=spec_id,
        source_file_path="test.mgf",
        mzs=np.array(mzs, dtype=np.float64),
        intensities=np.array(intensities, dtype=np.float64),
        precursor_mass=precursor_mass,
        charge=charge,
    )


def test_select_by_relative_intensity():
    assert select_by_relative_intensity(None) is None
    spec = make_spectrum([100.0, 200.0, 300.0], [1.0, 100.0, 0.05])
    filtered = select_by_relative_intensity(spec, intensity_from=0.001)
    assert filtered is not None
    assert len(filtered.mzs) == 2
    assert set(filtered.mzs) == {100.0, 200.0}


def test_reduce_to_number_of_peaks():
    assert reduce_to_number_of_peaks(None) is None
    spec = make_spectrum([100.0, 200.0, 300.0], [10.0, 50.0, 30.0])
    filtered = reduce_to_number_of_peaks(spec, n_max=2)
    assert filtered is not None
    assert len(filtered.mzs) == 2
    assert set(filtered.mzs) == {200.0, 300.0}


def test_require_minimum_number_of_high_peaks():
    assert require_minimum_number_of_high_peaks(None) is None
    spec = make_spectrum([100.0, 200.0, 300.0, 400.0], [100.0, 50.0, 10.0, 1.0])
    assert require_minimum_number_of_high_peaks(spec, no_peaks=3, intensity_percent=2.0) is not None
    assert require_minimum_number_of_high_peaks(spec, no_peaks=4, intensity_percent=2.0) is None


def test_remove_peaks_relative_to_precursor_mz_charge_aware_none():
    assert remove_peaks_relative_to_precursor_mz_charge_aware(None) is None


def test_remove_peaks_relative_to_precursor_mz_charge_aware_charge_1():
    spec = make_spectrum(
        [100.0, 498.0, 499.0, 500.0, 600.0],
        [10.0, 10.0, 10.0, 10.0, 10.0],
        precursor_mass=500.0,
        charge=1,
    )
    filtered = remove_peaks_relative_to_precursor_mz_charge_aware(spec)
    assert filtered is not None
    assert len(filtered.mzs) == 2
    assert set(filtered.mzs) == {100.0, 498.0}


def test_remove_peaks_relative_to_precursor_mz_charge_aware_charge_2():
    spec = make_spectrum(
        [100.0, 500.0, 600.0, 997.0, 998.0, 1000.0],
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        precursor_mass=500.0,
        charge=2,
    )
    filtered = remove_peaks_relative_to_precursor_mz_charge_aware(spec)
    assert filtered is not None
    assert len(filtered.mzs) == 4
    assert set(filtered.mzs) == {100.0, 500.0, 600.0, 997.0}


def test_keep_highest_intensity_in_radius_none():
    assert keep_highest_intensity_in_radius(None) is None


def test_keep_highest_intensity_in_radius_invalid_radius():
    spec = make_spectrum([100.0], [10.0])
    with pytest.raises(ValueError, match="Radius must be a positive number"):
        keep_highest_intensity_in_radius(spec, radius=0.0)


def test_keep_highest_intensity_in_radius_filtering():
    spec = make_spectrum(
        [10.0, 10.2, 10.4, 11.5, 11.9],
        [5.0, 10.0, 3.0, 20.0, 8.0],
    )
    filtered = keep_highest_intensity_in_radius(spec, radius=0.5)
    assert filtered is not None
    assert len(filtered.mzs) == 2
    assert set(filtered.mzs) == {10.2, 11.5}
