from kfp import dsl

from components.drift_collect_logs import collect_inference_logs
from components.drift_build_baseline import build_baseline_window
from components.drift_build_recent import build_recent_window
from components.drift_compute import compute_drift
from components.drift_alert import send_drift_alert
from components.lora_fine_llm import fine_tune_lora
from components.merge_for_serving import merge_for_serving
from components.serving import deploy_inferenceservice


@dsl.pipeline(name="qwen-csv-qa-drift-retrain")
def qwen_csv_qa_drift_retrain_pipeline(
    namespace: str = "mlops",
    model_name: str = "qwen-csv-qa",
    model_version: str = "v1",
    pvc_name: str = "model-store-pvc",
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    logs_root: str = "/mnt/models/qwen-csv-qa/prod-logs",
    window_hours: int = 24,
    webhook_url: str = "",
    max_seq_len: int = 512,
    batch_size: int = 1,
    grad_acc: int = 8,
    lr: float = 1e-4,
    epochs: int = 1,
):
    base_dir = f"/mnt/models/{model_name}/{model_version}"
    data_dir = f"{base_dir}/data"
    monitor_dir = f"{base_dir}/monitoring"
    adapter_dir = f"{base_dir}/adapter"
    serving_model_dir = f"{base_dir}/serving-model"

    recent_jsonl = f"{monitor_dir}/recent_window.jsonl"
    baseline_stats = f"{monitor_dir}/baseline_stats.json"
    recent_stats = f"{monitor_dir}/recent_stats.json"

    collect = collect_inference_logs(
        logs_root=logs_root,
        recent_jsonl_path=recent_jsonl,
        window_hours=window_hours,
    )

    baseline = build_baseline_window(
        baseline_stats_path=baseline_stats,
        bootstrap_jsonl=f"{data_dir}/finetune_val.jsonl",
        min_records=200,
    )
    baseline.after(collect)

    recent = build_recent_window(
        recent_jsonl_path=recent_jsonl,
        recent_stats_path=recent_stats,
    )
    recent.after(baseline)

    drift = compute_drift(
        baseline_stats_path=baseline_stats,
        recent_stats_path=recent_stats,
    )
    drift.after(recent)

    alert = send_drift_alert(
        drift_result_json=drift.output,
        webhook_url=webhook_url,
    )
    alert.after(drift)

    with dsl.If(drift.output.contains('"drift_detected": "true"')):
        train = fine_tune_lora(
            model_id=model_id,
            train_jsonl=f"{data_dir}/finetune_train.jsonl",
            val_jsonl=f"{data_dir}/finetune_val.jsonl",
            adapter_output_dir=adapter_dir,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            grad_acc=grad_acc,
            lr=lr,
            epochs=epochs,
        )

        merged = merge_for_serving(
            base_model_id=model_id,
            adapter_dir=f"{adapter_dir}/final",
            merged_output_dir=serving_model_dir,
        )
        merged.after(train)

        deploy = deploy_inferenceservice(
            namespace=namespace,
            model_name=model_name,
            runtime_name="kserve-huggingfaceserver",
            storage_uri=f"pvc://{pvc_name}/{model_name}/{model_version}/serving-model",
            cpu_request="4",
            cpu_limit="8",
            mem_request="16Gi",
            mem_limit="24Gi",
        )
        deploy.after(merged)
