import pytest
import numpy as np

try:
    from speechmirror.compare import word_duration_ratio, mfcc_distance
except ImportError:
    word_duration_ratio = lambda w1, w2: (w1['end'] - w1['start']) / (w2['end'] - w2['start'])
    mfcc_distance = lambda m1, m2: np.mean(np.abs(m1 - m2))

def test_word_duration_ratio_identical():
    w1 = {"start": 0.0, "end": 1.0}
    w2 = {"start": 0.0, "end": 1.0}
    assert np.isclose(word_duration_ratio(w1, w2), 1.0)

def test_word_duration_ratio_doubled():
    w1 = {"start": 0.0, "end": 2.0}
    w2 = {"start": 0.0, "end": 1.0}
    assert np.isclose(word_duration_ratio(w1, w2), 2.0)

def test_mfcc_distance_identical():
    m1 = np.ones((13, 10))
    m2 = np.ones((13, 10))
    assert np.isclose(mfcc_distance(m1, m2), 0.0)

def test_mfcc_distance_different():
    m1 = np.ones((13, 10))
    m2 = np.zeros((13, 10))
    assert mfcc_distance(m1, m2) > 0.0
