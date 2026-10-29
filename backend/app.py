"""
Flask API for the churn prediction tool.

Run locally:   python app.py         (serves http://localhost:5000)
Production:    gunicorn app:app
"""
from __future__ import annotations

import io
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

import pipeline as pl

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap
CORS(app)

WORK_DIR = Path(tempfile.gettempdir()) / "churn_runs"
WORK_DIR.mkdir(exist_ok=True)
RUN_TTL_SECONDS = 60 * 60
_lock = threading.Lock()


def _purge_old_runs():
    now = time.time()
    for folder in WORK_DIR.iterdir():
        try:
            if now - folder.stat().st_mtime > RUN_TTL_SECONDS:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


def _error(message: str, status: int = 400):
    return jsonify({"error": message}), status


@app.errorhandler(413)
def too_large(_e):
    return _error("The file is larger than 25 MB.", 413)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/preview")
def preview():
    """Read the upload and suggest which column is the churn label."""
    file = request.files.get("file")
    if file is None or not file.filename:
        return _error("Choose a CSV file first.")
    try:
        df = pl.read_csv_bytes(file.read())
    except ValueError as exc:
        return _error(str(exc))
    return jsonify(pl.to_py({
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "binary_columns": pl.binary_columns(df),
        "suggested_target": pl.detect_target(df),
        "head": {"columns": list(df.columns), "rows": df.head(5).values.tolist()},
    }))


@app.post("/api/analyze")
def analyze():
    """Run the full pipeline and return every result in one response."""
    file = request.files.get("file")
    target = request.form.get("target", "").strip()
    if file is None or not file.filename:
        return _error("Choose a CSV file first.")
    try:
        raw = pl.read_csv_bytes(file.read())
        target = target or pl.detect_target(raw) or ""
        cleaned, id_series, cleaning = pl.clean_dataset(raw, target)
        eda = pl.run_eda(cleaned, target)
        models, bundle = pl.train_models(cleaned, target)
        scored, scoring = pl.score_customers(cleaned, id_series, target, bundle)
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # unexpected: keep the server alive, tell the user plainly
        app.logger.exception("analysis failed")
        return _error(f"Something went wrong while analysing the file ({type(exc).__name__}). "
                      "Check that the file is a customer churn CSV.", 500)

    with _lock:
        _purge_old_runs()
    run_id = uuid.uuid4().hex
    run_dir = WORK_DIR / run_id
    run_dir.mkdir()
    cleaned.to_csv(run_dir / "cleaned.csv", index=False)
    scored.to_csv(run_dir / "scored.csv", index=False)
    pd.DataFrame(models["comparison"]["rows"]).to_csv(run_dir / "model_comparison.csv", index=False)

    return jsonify({
        "run_id": run_id,
        "filename": file.filename,
        "cleaning": cleaning,
        "eda": eda,
        "model_building": models["model_building"],
        "comparison": models["comparison"],
        "evaluation": models["evaluation"],
        "scoring": scoring,
    })


@app.get("/api/download/<run_id>/<kind>")
def download(run_id: str, kind: str):
    names = {"cleaned": "cleaned_dataset.csv", "scored": "churn_scored_customers.csv",
             "comparison": "model_comparison.csv"}
    source = {"cleaned": "cleaned.csv", "scored": "scored.csv", "comparison": "model_comparison.csv"}
    if kind not in names or not run_id.isalnum():
        return _error("Unknown download.", 404)
    path = WORK_DIR / run_id / source[kind]
    if not path.exists():
        return _error("This result has expired. Upload the file again to rebuild it.", 404)
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name=names[kind])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
