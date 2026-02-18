from kfp import dsl
from kfp import kubernetes

from components.generate_instructions_data import prepare_instruction_data
from components.lora_fine_llm import fine_tune_lora
from components.merge_for_serving import merge_for_serving
from components.model_registry import register_model_version
from components.serving import deploy_inferenceservice


@dsl.component(base_image="python:3.11", packages_to_install=[])
def evaluate_dataset_only(val_jsonl: str, eval_out: dsl.OutputPath(str)):
    import json
    from pathlib import Path

    total = 0
    unknown = 0
    bad = 0
    with open(val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                row = json.loads(line)
                if row.get("answer") == "Not available in provided data.":
                    unknown += 1
            except Exception:
                bad += 1

    metrics = {
        "val_rows": total,
        "invalid_rows": bad,
        "unknown_answer_rows": unknown,
        "unknown_ratio": (unknown / total) if total else 0.0,
    }
    Path(eval_out).write_text(json.dumps(metrics), encoding="utf-8")


@dsl.pipeline(
    name="qwen-csv-qa-train-and-serve-v2",
    description="Generate instruction data, fine-tune Qwen LoRA on GPU, and deploy to KServe endpoint.",
)
def qwen_csv_qa_pipeline(
    namespace: str = "mlops",
    csv_path: str = "/mnt/models/data/LimeSpark Phase2_2026-01-23-1416.csv",
    pvc_name: str = "model-store-pvc",
    model_name: str = "qwen-csv-qa",
    model_version: str = "v1",
    data_version: str = "d2026-02-13",
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    target_rows: int = 5000,
    val_ratio: float = 0.1,
    seed: int = 42,
    max_seq_len: int = 512,
    batch_size: int = 1,
    grad_acc: int = 8,
    lr: float = 0.0001,
    epochs: int = 1,
    registry_server: str = "http://model-registry-service.kubeflow.svc.cluster.local:8080",
    deploy_enabled: bool = True,
):
    base_dir = f"/mnt/models/{model_name}/{model_version}"
    data_dir = f"{base_dir}/data/{data_version}"
    adapter_dir = f"{base_dir}/adapter/{data_version}"
    merged_dir = f"{base_dir}/serving-model"
    storage_uri = f"pvc://{pvc_name}/{model_name}/{model_version}/serving-model"

    prep = prepare_instruction_data(
        csv_path=csv_path,
        output_dir=data_dir,
        target_rows=target_rows,
        val_ratio=val_ratio,
        seed=seed,
    )
    kubernetes.mount_pvc(prep, pvc_name=pvc_name, mount_path="/mnt/models")
    prep.set_cpu_request("2")
    prep.set_cpu_limit("4")
    prep.set_memory_request("8Gi")
    prep.set_memory_limit("12Gi")

    train = fine_tune_lora(
        train_jsonl=prep.outputs["train_jsonl_out"],
        val_jsonl=prep.outputs["val_jsonl_out"],
        adapter_output_dir=adapter_dir,
        model_id=model_id,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        grad_acc=grad_acc,
        lr=lr,
        epochs=epochs,
    )
    kubernetes.mount_pvc(train, pvc_name=pvc_name, mount_path="/mnt/models")
    kubernetes.add_node_selector(train, label_key="agentpool", label_value="gpuac3d")
    kubernetes.add_node_selector(train, label_key="kubernetes.azure.com/accelerator", label_value="nvidia")
    kubernetes.add_toleration(train, key="type", operator="Equal", value="gpu", effect="NoSchedule")
    train.set_cpu_request("8")
    train.set_cpu_limit("12")
    train.set_memory_request("48Gi")
    train.set_memory_limit("64Gi")
    train.set_accelerator_type("nvidia.com/gpu")
    train.set_accelerator_limit(1)
    kubernetes.use_secret_as_env(
        train,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
    )
    train.set_caching_options(True)

    evaluate = evaluate_dataset_only(val_jsonl=prep.outputs["val_jsonl_out"])
    kubernetes.mount_pvc(evaluate, pvc_name=pvc_name, mount_path="/mnt/models")
    evaluate.set_cpu_request("2")
    evaluate.set_cpu_limit("4")
    evaluate.set_memory_request("4Gi")
    evaluate.set_memory_limit("8Gi")
    evaluate.after(train)

    merge = merge_for_serving(
        base_model_id=model_id,
        adapter_dir=f"{adapter_dir}/final",
        merged_output_dir=merged_dir,
    )
    kubernetes.mount_pvc(merge, pvc_name=pvc_name, mount_path="/mnt/models")
    kubernetes.use_secret_as_env(
        merge,
        secret_name="hf-token",
        secret_key_to_env={"HF_TOKEN": "HF_TOKEN"},
    )
    kubernetes.add_node_selector(merge, label_key="agentpool", label_value="gpuac3d")
    kubernetes.add_node_selector(merge, label_key="kubernetes.azure.com/accelerator", label_value="nvidia")
    kubernetes.add_toleration(merge, key="type", operator="Equal", value="gpu", effect="NoSchedule")
    merge.set_cpu_request("4")
    merge.set_cpu_limit("8")
    merge.set_memory_request("16Gi")
    merge.set_memory_limit("24Gi")
    merge.set_accelerator_type("nvidia.com/gpu")
    merge.set_accelerator_limit(1)
    merge.set_caching_options(True)
    merge.after(evaluate)

    register = register_model_version(
        registry_server=registry_server,
        model_name=model_name,
        model_version=model_version,
        storage_uri=storage_uri,
        model_id=model_id,
        description="Qwen CSV QA LoRA model registered from KFP pipeline.",
    )
    register.set_cpu_request("500m")
    register.set_cpu_limit("1")
    register.set_memory_request("1Gi")
    register.set_memory_limit("2Gi")
    register.after(merge)
    register.set_caching_options(False)

    with dsl.If(deploy_enabled == True):
        deploy = deploy_inferenceservice(
            model_name=model_name,
            namespace=namespace,
            storage_uri=storage_uri,
            runtime_name="kserve-huggingfaceserver",
            cpu_request="4",
            cpu_limit="8",
            mem_request="16Gi",
            mem_limit="24Gi",
        )
        deploy.after(register)
        deploy.set_caching_options(False)
