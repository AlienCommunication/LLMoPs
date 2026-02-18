# mlops-labs

Kubeflow Pipelines v2:

1. Building instruction/QA data from CSV.
2. QLoRA fine-tuning (Qwen2.5-1.5B-Instruct) on T4 GPU.
3. Evaluation gate.
4. KServe endpoint deployment from PVC storage URI.

## Directory

- `components/generate_instructions_data.py`: CSV -> train/val instruction dataset generation.
- `components/lora_fine_llm.py`: QLoRA training code.
- `components/serving.py`: KServe `InferenceService` build/apply/wait helpers.
- `pipeline/kfp_pipeline.py`: KFP v2 pipeline and component scheduling/resources.
- `pipeline/compile.py`: compiles pipeline to YAML package.
- `kubeflow-resources/inferenceservice-qwen-csv-qa.yaml`: standalone deployment manifest.

## Infra assumptions

- Namespace: `mlops`
- PVC: `model-store-pvc` (RWX, Azure Files)
- GPU nodepool labels:
  - `agentpool=gpuac3d`
  - `kubernetes.azure.com/accelerator=nvidia`
- GPU toleration:
  - `type=gpu:NoSchedule`
- KServe runtime: `kserve-huggingfaceserver`

## Compile pipeline, Use this if you want to compile locally and upload compiled yaml to kubeflow dashboard and create run

```bash
pip install -r requirements.txt
python pipeline/compile.py
```

Generated package: `pipeline/qwen_csv_qa_pipeline.yaml`

## Run parameters (important)

- `csv_path`: CSV location mounted in pipeline pods (default `/mnt/models/data/...csv`).
- `pvc_name`: defaults `model-store-pvc`.
- `model_name`, `model_version`: control artifact path and endpoint storage URI.

Artifacts are written to:

- `/mnt/models/<model_name>/<model_version>/data`
- `/mnt/models/<model_name>/<model_version>/adapter/final`

KServe storage URI used by pipeline:

- `pvc://<pvc_name>/<model_name>/<model_version>/adapter/final`

## Notes
- For spot-GPU scheduling, add a pipeline variant with `agentpool=gpuspot13ca` and spot tolerations.


If your environment needs explicit KFP endpoint:

```bash
export KFP_HOST="https://<your-kfp-endpoint>/pipeline"
export KFP_NAMESPACE="mlops"
python pipeline/submit_run.py
```


## GitHub Actions CI/CD (No Prometej)

This repo includes native KFP CI/CD at `.github/workflows/kfp-cicd.yml`.

Required GitHub secrets:

- `KFP_HOST`: Kubeflow Pipelines API host (e.g. `http://ml-pipeline.kubeflow.svc.cluster.local:8888` for in-network runner)
- `KFP_TOKEN`: Kubeflow service account token

Workflow behavior:

1. Compiles `pipeline.kfp_pipeline:qwen_csv_qa_pipeline`
2. Creates or updates pipeline in Kubeflow
3. Optionally creates a run (default: true for workflow_dispatch)

