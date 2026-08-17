from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_probes(run_dir: str | Path, position: str = "panl") -> list[dict[str, Any]]:
    import numpy as np
    from safetensors.torch import load_file
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    run_dir = Path(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines() if line.strip()]
    records = [row for row in records if isinstance(row.get("correct_exact_match"), bool)]
    first = load_file(str(run_dir / records[0]["activation_file"]))
    keys = sorted(key for key in first if key.startswith(position + ".layer_"))
    correctness = np.asarray([int(row["correct_exact_match"]) for row in records])
    confidence = np.asarray([float(row["confidence_midpoint"]) for row in records])
    results = []
    for key in keys:
        x = np.stack([load_file(str(run_dir / row["activation_file"]))[key].numpy() for row in records])
        classification = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", max_iter=2000))
        pred_correct = cross_val_predict(classification, x, correctness, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
        regression = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pred_confidence = cross_val_predict(regression, x, confidence, cv=KFold(5, shuffle=True, random_state=42))
        results.append({
            "key": key,
            "correctness_auroc": float(roc_auc_score(correctness, pred_correct)),
            "confidence_r2": float(r2_score(confidence, pred_confidence)),
        })
    return results
