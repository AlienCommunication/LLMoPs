from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[],
)
def compute_drift(
    baseline_stats_path: str,
    recent_stats_path: str,
    unknown_rate_delta_threshold: float = 0.08,
    answer_len_delta_threshold: float = 3.0,
    product_jsd_threshold: float = 0.20,
    missing_rate_delta_threshold: float = 0.15,
) -> str:
    import json
    import math

    def js_divergence(p, q):
        keys = set(p.keys()) | set(q.keys())
        if not keys:
            return 0.0
        eps = 1e-12
        m = {}
        for k in keys:
            m[k] = 0.5 * (p.get(k, 0.0) + q.get(k, 0.0))

        def kl(a, b):
            s = 0.0
            for k in keys:
                ak = max(a.get(k, 0.0), eps)
                bk = max(b.get(k, 0.0), eps)
                s += ak * math.log(ak / bk)
            return s

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    with open(baseline_stats_path, "r", encoding="utf-8") as f:
        b = json.load(f)
    with open(recent_stats_path, "r", encoding="utf-8") as f:
        r = json.load(f)

    if r.get("n", 0) == 0:
        out = {
            "drift_detected": "false",
            "reason": "recent_window_empty",
            "details": {},
        }
        return json.dumps(out, ensure_ascii=True)

    unknown_delta = abs(r["unknown_rate"] - b["unknown_rate"])
    answer_len_delta = abs(r["answer_len_avg"] - b["answer_len_avg"])
    jsd = js_divergence(b.get("product_code_dist", {}), r.get("product_code_dist", {}))

    fields = set(b.get("missing_rate_by_field", {}).keys()) | set(r.get("missing_rate_by_field", {}).keys())
    max_missing_delta = 0.0
    worst_field = ""
    for f in fields:
        d = abs(r.get("missing_rate_by_field", {}).get(f, 0.0) - b.get("missing_rate_by_field", {}).get(f, 0.0))
        if d > max_missing_delta:
            max_missing_delta = d
            worst_field = f

    checks = {
        "unknown_rate_drift": unknown_delta > unknown_rate_delta_threshold,
        "answer_len_drift": answer_len_delta > answer_len_delta_threshold,
        "product_mix_drift": jsd > product_jsd_threshold,
        "missing_field_drift": max_missing_delta > missing_rate_delta_threshold,
    }

    drift_detected = any(checks.values())

    out = {
        "drift_detected": "true" if drift_detected else "false",
        "reason": "threshold_exceeded" if drift_detected else "within_threshold",
        "details": {
            "unknown_delta": unknown_delta,
            "answer_len_delta": answer_len_delta,
            "product_jsd": jsd,
            "max_missing_delta": max_missing_delta,
            "worst_field": worst_field,
            "checks": checks,
            "baseline_n": b.get("n", 0),
            "recent_n": r.get("n", 0),
        },
    }
    return json.dumps(out, ensure_ascii=True)
