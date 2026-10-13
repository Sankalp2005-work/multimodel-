"""
scripts/analyze_errors.py -- Audit classification errors on sealed test set.
Inspects misclassified clips in results/test_report.json across real and synthetic splits.
"""
import json
from pathlib import Path

rep_path = Path("results/test_report.json")
with open(rep_path, "r", encoding="utf-8") as f:
    rep = json.load(f)

preds = rep.get("clip_predictions", [])
incorrect = [p for p in preds if p["true_type"] != p["pred_type"]]

print(f"Total test samples: {len(preds)}")
print(f"Total correct: {len(preds) - len(incorrect)}")
print(f"Total incorrect: {len(incorrect)}\n")

for p in incorrect:
    sid = p["sample_id"]
    stype = p["source_type"]
    ttype = p["true_type"]
    tsev = p["true_sev"]
    ptype = p["pred_type"]
    psev = p["pred_sev"]
    conf = p.get("pred_conf", 0)
    print(f"[{sid}] ({stype}) True: {ttype} (sev {tsev}) | Pred: {ptype} (sev {psev}, conf {conf:.2f})")
