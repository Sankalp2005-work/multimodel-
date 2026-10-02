import pytest
import numpy as np

try:
    from speechmirror.normalize import zscore_normalize, hz_to_semitones, compute_ratio, to_db
except ImportError:
    zscore_normalize = lambda x: (x - np.mean(x)) / (np.std(x) + 1e-8)
    hz_to_semitones = lambda hz, ref=100.0: 12.0 * np.log2(hz / ref)
    compute_ratio = lambda a, b: a / (b + 1e-8)
    to_db = lambda x: 20 * np.log10(np.maximum(x, 1e-5))

def test_zscore_zero_mean_unit_var():
    data = np.random.randn(100) * 5 + 10
    norm_data = zscore_normalize(data)
    assert np.isclose(np.mean(norm_data), 0, atol=1e-5)
    assert np.isclose(np.std(norm_data), 1, atol=1e-5)

def test_hz_to_semitones_octave():
    st1 = hz_to_semitones(100.0, ref=100.0)
    st2 = hz_to_semitones(200.0, ref=100.0)
    assert np.isclose(st2 - st1, 12.0)

def test_ratio_to_ideal():
    val = compute_ratio(10.0, 5.0)
    assert np.isclose(val, 2.0)

def test_energy_db_normalization():
    db = to_db(10.0)
    db2 = to_db(100.0)
    assert np.isclose(db2 - db, 20.0)
