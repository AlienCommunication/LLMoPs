from __future__ import annotations

import sys
from pathlib import Path

from kfp import compiler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.kfp_pipeline import qwen_csv_qa_pipeline


if __name__ == "__main__":
    out = ROOT / "pipeline" / "qwen_csv_qa_pipeline.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=qwen_csv_qa_pipeline,
        package_path=str(out),
    )
    print(f"Wrote pipeline package: {out}")
