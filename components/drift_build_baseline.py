from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[],
)
def build_baseline_window(
    baseline_stats_path: str,
    bootstrap_jsonl: str,
    min_records: int = 200,
) -> str:
    import json
    import math
    import os
    from collections import Counter

    def compute_stats(path: str):
        unknown_text = "Not available in provided data."
        n = 0
        unknown = 0
        answer_lens = []
        product_codes = Counter()
        missing_fields = Counter()
        context_seen = Counter()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue

                answer = r.get("answer") or r.get("response") or ""
                context = r.get("context", {}) or {}

                n += 1
                if answer.strip() == unknown_text:
                    unknown += 1
                answer_lens.append(len(answer.split()))

                pc = str(context.get("PRODUCT_CODE", "")).strip()
                if pc:
                    product_codes[pc] += 1

                for k, v in context.items():
                    context_seen[k] += 1
                    s = str(v).strip().upper() if v is not None else ""
                    if s in {"", "UNKNOWN", "N/A", "NA", "NULL", "NONE", "UNASSIGNED"}:
                        missing_fields[k] += 1

        if n == 0:
            return {
                "n": 0,
                "unknown_rate": 0.0,
                "answer_len_avg": 0.0,
                "answer_len_std": 0.0,
                "product_code_dist": {},
                "missing_rate_by_field": {},
            }

        mean_len = sum(answer_lens) / n
        var = sum((x - mean_len) ** 2 for x in answer_lens) / n
        std = math.sqrt(var)

        pc_total = sum(product_codes.values()) or 1
        pc_dist = {k: v / pc_total for k, v in product_codes.items()}

        miss_rate = {}
        for k, seen in context_seen.items():
            miss_rate[k] = (missing_fields[k] / seen) if seen else 0.0

        return {
            "n": n,
            "unknown_rate": unknown / n,
            "answer_len_avg": mean_len,
            "answer_len_std": std,
            "product_code_dist": pc_dist,
            "missing_rate_by_field": miss_rate,
        }

    os.makedirs(os.path.dirname(baseline_stats_path), exist_ok=True)

    if os.path.exists(baseline_stats_path):
        with open(baseline_stats_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        return json.dumps({"baseline_status": "reused", "n": baseline.get("n", 0)})

    stats = compute_stats(bootstrap_jsonl)
    if stats["n"] < min_records:
        raise ValueError(f"Not enough records to create baseline. got={stats['n']} min={min_records}")

    with open(baseline_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=True, indent=2)

    return json.dumps({"baseline_status": "created", "n": stats["n"]})
