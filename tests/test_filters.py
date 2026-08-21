"""Tests for custom matchms filters."""

import numpy as np
import pytest
from matchms import Spectrum

from kakapo.utils.filters import (
    keep_highest_intensity_in_radius,
    remove_peaks_relative_to_precursor_mz_charge_aware,
)


def test_remove_peaks_relative_to_precursor_mz_charge_aware_none():
    """Should return None if the input spectrum is None."""
    assert remove_peaks_relative_to_precursor_mz_charge_aware(None) is None


def test_remove_peaks_relative_to_precursor_mz_charge_aware_no_precursor():
    """Should raise ValueError if precursor_mz is undefined."""
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([10.0, 20.0], dtype="float"),
        metadata={"charge": 1},
    )
    with pytest.raises(ValueError, match="Undefined 'precursor_mz'"):
        remove_peaks_relative_to_precursor_mz_charge_aware(spec)


def test_remove_peaks_relative_to_precursor_mz_charge_aware_invalid_precursor():
    """Should raise ValueError if precursor_mz is not a scalar number."""
    spec = Spectrum(
        mz=np.array([100.0, 200.0], dtype="float"),
        intensities=np.array([10.0, 20.0], dtype="float"),
        metadata={"precursor_mz": 500.0, "charge": 1},
    )
    spec.set("precursor_mz", "invalid_mz")
    with pytest.raises(ValueError, match="Expected 'precursor_mz' to be a scalar number"):
        remove_peaks_relative_to_precursor_mz_charge_aware(spec)


def test_remove_peaks_relative_to_precursor_mz_charge_aware_charge_1():
    """Should correctly filter peaks for charge=1."""
    # Precursor m/z = 500.0. Max fragment m/z = 500.0.
    # With offset = -1.6, cutoff = 498.4.
    spec = Spectrum(
        mz=np.array([100.0, 498.0, 499.0, 500.0, 600.0], dtype="float"),
        intensities=np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype="float"),
        metadata={"precursor_mz": 500.0, "charge": 1},
    )
    filtered = remove_peaks_relative_to_precursor_mz_charge_aware(spec)
    assert filtered is not None
    assert len(filtered.peaks) == 2
    assert np.allclose(filtered.peaks.mz, [100.0, 498.0])


def test_remove_peaks_relative_to_precursor_mz_charge_aware_charge_2():
    """Should correctly filter peaks for charge=2, keeping peaks above precursor_mz."""
    # Precursor m/z = 500.0. Charge = 2.
    # PROTON_MASS = 1.007276.
    # Max fragment m/z = 500.0 * 2 - (2 - 1) * 1.007276 = 1000.0 - 1.007276 = 998.992724.
    # With offset = -1.6, cutoff = 997.392724.
    spec = Spectrum(
        mz=np.array([100.0, 500.0, 600.0, 997.0, 998.0, 1000.0], dtype="float"),
        intensities=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype="float"),
        metadata={"precursor_mz": 500.0, "charge": 2},
    )
    filtered = remove_peaks_relative_to_precursor_mz_charge_aware(spec)
    assert filtered is not None
    # Cutoff is 997.392724. So peaks 100.0, 500.0, 600.0, 997.0 should be kept.
    # 998.0 and 1000.0 should be filtered out.
    assert len(filtered.peaks) == 4
    assert np.allclose(filtered.peaks.mz, [100.0, 500.0, 600.0, 997.0])


def test_remove_peaks_relative_to_precursor_mz_charge_aware_custom_offset():
    """Should honor custom offset_to_precursor parameter."""
    # Precursor m/z = 500.0. Charge = 1.
    # Custom offset = 5.0. Cutoff = 505.0.
    spec = Spectrum(
        mz=np.array([100.0, 498.0, 502.0, 506.0], dtype="float"),
        intensities=np.array([10.0, 10.0, 10.0, 10.0], dtype="float"),
        metadata={"precursor_mz": 500.0, "charge": 1},
    )
    filtered = remove_peaks_relative_to_precursor_mz_charge_aware(spec, offset_to_precursor=5.0)
    assert filtered is not None
    assert len(filtered.peaks) == 3
    assert np.allclose(filtered.peaks.mz, [100.0, 498.0, 502.0])


def test_remove_peaks_relative_to_precursor_mz_charge_aware_clone():
    """Should correctly clone or modify in-place based on the clone parameter."""
    spec = Spectrum(
        mz=np.array([100.0, 600.0], dtype="float"),
        intensities=np.array([10.0, 10.0], dtype="float"),
        metadata={"precursor_mz": 500.0, "charge": 1},
    )
    # Clone = True
    filtered_cloned = remove_peaks_relative_to_precursor_mz_charge_aware(spec, clone=True)
    assert filtered_cloned is not None
    assert filtered_cloned is not spec
    assert len(spec.peaks) == 2  # Original remains unchanged
    assert len(filtered_cloned.peaks) == 1

    # Clone = False
    filtered_inplace = remove_peaks_relative_to_precursor_mz_charge_aware(spec, clone=False)
    assert filtered_inplace is not None
    assert filtered_inplace is spec
    assert len(spec.peaks) == 1  # Original modified in-place


def test_keep_highest_intensity_in_radius_none():
    """Should return None if the input spectrum is None."""
    assert keep_highest_intensity_in_radius(None) is None


def test_keep_highest_intensity_in_radius_empty():
    """Should return the spectrum unmodified if it has no peaks."""
    spec = Spectrum(
        mz=np.array([], dtype="float"),
        intensities=np.array([], dtype="float"),
        metadata={"precursor_mz": 500.0},
    )
    filtered = keep_highest_intensity_in_radius(spec)
    assert filtered is not None
    assert len(filtered.peaks) == 0


def test_keep_highest_intensity_in_radius_invalid_radius():
    """Should raise ValueError if radius is less than or equal to zero."""
    spec = Spectrum(
        mz=np.array([100.0], dtype="float"),
        intensities=np.array([10.0], dtype="float"),
        metadata={"precursor_mz": 500.0},
    )
    with pytest.raises(ValueError, match="Radius must be a positive number"):
        keep_highest_intensity_in_radius(spec, radius=0.0)

    with pytest.raises(ValueError, match="Radius must be a positive number"):
        keep_highest_intensity_in_radius(spec, radius=-0.5)


def test_keep_highest_intensity_in_radius_filtering():
    """Should keep a peak only if it has the highest intensity in its window."""
    # Peak 10.0: window [9.5, 10.5], peaks: 10.0(5), 10.2(10), 10.4(3). Max 10 -> Discard.
    # Peak 10.2: window [9.7, 10.7], peaks: 10.0(5), 10.2(10), 10.4(3). Max 10 -> Keep.
    # Peak 10.4: window [9.9, 10.9], peaks: 10.0(5), 10.2(10), 10.4(3). Max 10 -> Discard.
    # Peak 11.5: window [11.0, 12.0], peaks: 11.5(20), 11.9(8). Max 20 -> Keep.
    # Peak 11.9: window [11.4, 12.4], peaks: 11.5(20), 11.9(8). Max 20 -> Discard.
    # Expected kept peaks: 10.2 (10.0) and 11.5 (20.0).
    spec = Spectrum(
        mz=np.array([10.0, 10.2, 10.4, 11.5, 11.9], dtype="float"),
        intensities=np.array([5.0, 10.0, 3.0, 20.0, 8.0], dtype="float"),
        metadata={"precursor_mz": 500.0},
    )

    filtered = keep_highest_intensity_in_radius(spec, radius=0.5)
    assert filtered is not None
    assert len(filtered.peaks) == 2
    assert np.allclose(filtered.peaks.mz, [10.2, 11.5])
    assert np.allclose(filtered.peaks.intensities, [10.0, 20.0])


def test_keep_highest_intensity_in_radius_custom_radius():
    """Should correctly apply custom radius for window calculations."""
    # With radius = 0.1, each peak has no other peaks within its window.
    # All peaks should be kept.
    spec = Spectrum(
        mz=np.array([10.0, 10.2, 10.4, 11.5, 11.9], dtype="float"),
        intensities=np.array([5.0, 10.0, 3.0, 20.0, 8.0], dtype="float"),
        metadata={"precursor_mz": 500.0},
    )

    filtered = keep_highest_intensity_in_radius(spec, radius=0.1)
    assert filtered is not None
    assert len(filtered.peaks) == 5
    assert np.allclose(filtered.peaks.mz, [10.0, 10.2, 10.4, 11.5, 11.9])


def test_keep_highest_intensity_in_radius_ties():
    """Should keep all peaks achieving the maximum intensity in case of a tie."""
    spec = Spectrum(
        mz=np.array([10.0, 10.2], dtype="float"),
        intensities=np.array([10.0, 10.0], dtype="float"),
        metadata={"precursor_mz": 500.0},
    )

    filtered = keep_highest_intensity_in_radius(spec, radius=0.5)
    assert filtered is not None
    assert len(filtered.peaks) == 2
    assert np.allclose(filtered.peaks.mz, [10.0, 10.2])


def test_keep_highest_intensity_in_radius_clone():
    """Should correctly clone or modify in-place based on the clone parameter."""
    spec = Spectrum(
        mz=np.array([10.0, 10.2], dtype="float"),
        intensities=np.array([5.0, 10.0], dtype="float"),
        metadata={"precursor_mz": 500.0},
    )

    # Clone = True
    filtered_cloned = keep_highest_intensity_in_radius(spec, radius=0.5, clone=True)
    assert filtered_cloned is not None
    assert filtered_cloned is not spec
    assert len(spec.peaks) == 2  # Original remains unchanged
    assert len(filtered_cloned.peaks) == 1

    # Clone = False
    filtered_inplace = keep_highest_intensity_in_radius(spec, radius=0.5, clone=False)
    assert filtered_inplace is not None
    assert filtered_inplace is spec
    assert len(spec.peaks) == 1  # Original modified in-place
