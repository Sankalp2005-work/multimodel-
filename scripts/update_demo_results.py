import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.speechmirror.pipeline import SpeechMirrorPipeline

def update_demo():
    pipe = SpeechMirrorPipeline()
    ideal_f = pipe.extract_features("data/demo/ideal_demo.wav")
    part_f = pipe.extract_features("data/demo/participant_demo.wav")
    wp = pipe.align_dtw(part_f, ideal_f)

    ideal_times = ideal_f["frame_times"]
    part_times = part_f["frame_times"]
    step = max(1, len(wp) // 150)
    warping_path = [
        [
            round(float(ideal_times[min(int(wp[i, 1]), len(ideal_times) - 1)]), 3),
            round(float(part_times[min(int(wp[i, 0]), len(part_times) - 1)]), 3)
        ]
        for i in range(0, len(wp), step)
    ]
    last_pt = [round(float(ideal_times[-1]), 3), round(float(part_times[-1]), 3)]
    if warping_path[-1] != last_pt:
        warping_path.append(last_pt)

    demo_path = Path("data/demo/demo_results.json")
    with open(demo_path, "r") as f:
        demo_data = json.load(f)

    demo_data["alignment"]["warping_path"] = warping_path

    transcript_words = [
        ("The", 0.15, 0.40, False, None),
        ("acoustic", 0.42, 0.95, False, None),
        ("analyzer", 0.98, 1.48, False, None),
        ("measures", 1.50, 1.95, True, "rushed_pace"),
        ("speech", 1.96, 2.38, True, "rushed_pace"),
        ("velocity", 2.40, 2.88, True, "rushed_pace"),
        ("across", 2.90, 3.38, True, "rushed_pace"),
        ("aligned", 3.42, 3.98, False, None),
        ("phoneme", 4.02, 4.65, False, None),
        ("intervals.", 4.68, 5.18, False, None),
        ("pause", 5.20, 7.50, True, "dead_pause"),
        ("Unexpected", 7.55, 8.25, False, None),
        ("pauses", 8.28, 8.92, False, None),
        ("disrupt", 9.00, 9.65, True, "flat_pitch"),
        ("listener", 9.68, 10.42, True, "flat_pitch"),
        ("comprehension,", 10.45, 12.18, True, "flat_pitch"),
        ("while", 12.22, 12.65, False, None),
        ("flat", 12.68, 13.05, False, None),
        ("cadence", 13.08, 13.48, False, None),
        ("suppresses", 13.50, 14.30, True, "mumbled_clarity"),
        ("articulation", 14.32, 15.35, True, "mumbled_clarity"),
        ("resonance.", 15.38, 16.78, True, "mumbled_clarity"),
    ]
    demo_data["words"] = [
        {
            "id": f"w{i+1}",
            "word": w,
            "start": s,
            "end": e,
            "isFlawed": is_f,
            "flawType": ft,
            "isPause": ft == "dead_pause"
        }
        for i, (w, s, e, is_f, ft) in enumerate(transcript_words)
    ]

    with open(demo_path, "w") as f:
        json.dump(demo_data, f, indent=2)

    print(f"Updated demo_results.json with {len(warping_path)} warping points and {len(demo_data['words'])} words.")

if __name__ == "__main__":
    update_demo()
