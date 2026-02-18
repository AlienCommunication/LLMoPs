from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[],
)
def collect_inference_logs(
    logs_root: str,
    recent_jsonl_path: str,
    window_hours: int = 24,
) -> str:
    import json
    import os
    from datetime import datetime, timedelta, timezone

    os.makedirs(os.path.dirname(recent_jsonl_path), exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    kept = 0

    with open(recent_jsonl_path, "w", encoding="utf-8") as out_f:
        for root, _, files in os.walk(logs_root):
            for fn in files:
                if not fn.endswith(".jsonl"):
                    continue
                fp = os.path.join(root, fn)
                with open(fp, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        ts = rec.get("ts")
                        if not ts:
                            continue
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        except Exception:
                            continue
                        if dt >= cutoff:
                            out_f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                            kept += 1

    return f"collected={kept};path={recent_jsonl_path}"
