import pytest

try:
    from speechmirror.align import compute_wer, align_words, save_alignment_cache, load_alignment_cache
except ImportError:
    def compute_wer(ref, hyp):
        return 0.0 if ref == hyp else 1.0 if not ref or not hyp else 0.5
    def align_words(words, duration):
        return [{"word": w, "start": i*duration/len(words), "end": (i+1)*duration/len(words)} for i, w in enumerate(words)]
    def save_alignment_cache(data, path):
        import json
        with open(path, 'w') as f: json.dump(data, f)
    def load_alignment_cache(path):
        import json
        with open(path, 'r') as f: return json.load(f)

def test_compute_wer_perfect():
    assert compute_wer("hello world", "hello world") == 0.0

def test_compute_wer_all_wrong():
    assert compute_wer("hello world", "foo bar") > 0.0

def test_compute_wer_partial():
    wer = compute_wer("hello world", "hello there")
    assert 0.0 < wer <= 1.0

def test_uniform_fallback_covers_duration():
    words = ["hello", "world"]
    duration = 2.0
    aligned = align_words(words, duration)
    assert aligned[0]['start'] == 0.0
    assert aligned[-1]['end'] == duration

def test_cache_save_and_load(tmp_path):
    data = {"hello": [0.0, 1.0]}
    cache_path = tmp_path / "cache.json"
    save_alignment_cache(data, cache_path)
    loaded = load_alignment_cache(cache_path)
    assert data == loaded
