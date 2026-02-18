import json
import os
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

TARGET_URL = os.getenv("TARGET_URL", "http://qwen-csv-qa.mlops.svc.cluster.local/v1/models/qwen-csv-qa:predict")
LOG_DIR = os.getenv("LOG_DIR", "/mnt/models/qwen-csv-qa/prod-logs")
LOG_FILE = os.getenv("LOG_FILE", "inference.jsonl")
TIMEOUT_SEC = float(os.getenv("TIMEOUT_SEC", "60"))

os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)


def append_log(record: dict):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200


@app.route("/predict", methods=["POST"])
def predict():
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    body = request.get_json(silent=True) or {}

    question = body.get("question", "")
    context = body.get("context", {})
    prompt = body.get("prompt", "")

    status = "ok"
    response_json = {}
    err = ""

    try:
        r = requests.post(TARGET_URL, json=body, timeout=TIMEOUT_SEC)
        response_json = r.json() if r.content else {}
        if r.status_code >= 400:
            status = "error"
            err = f"http_{r.status_code}"
    except Exception as e:
        status = "error"
        err = str(e)

    answer = ""
    if isinstance(response_json, dict):
        preds = response_json.get("predictions")
        if isinstance(preds, list) and preds:
            answer = str(preds[0])

    append_log(
        {
            "ts": ts,
            "status": status,
            "error": err,
            "question": question,
            "context": context,
            "prompt": prompt,
            "answer": answer,
            "raw_response": response_json,
        }
    )

    if status == "ok":
        return jsonify(response_json), 200
    return jsonify({"error": err, "raw_response": response_json}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
