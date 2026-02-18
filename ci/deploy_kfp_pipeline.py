import argparse
import datetime as dt
import json
import os
from pathlib import Path

import kfp
from kfp import compiler


def read_token() -> str:
    env_token = os.getenv("KFP_TOKEN", "").strip()
    if env_token:
        return env_token

    sa_token_path = Path("/var/run/secrets/kubeflow/pipelines/token")
    if sa_token_path.exists():
        return sa_token_path.read_text(encoding="utf-8").strip()

    raise RuntimeError(
        "No KFP token found. Set KFP_TOKEN env var or run inside cluster with service account token file."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy KFP pipeline and optionally create a run")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pipeline-name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--pipeline-func", default="pipeline.kfp_pipeline:qwen_csv_qa_pipeline")
    parser.add_argument("--pipeline-yaml", default="pipeline/qwen_csv_qa_pipeline.yaml")
    parser.add_argument("--experiment-name", default="qwen-csv-qa-experiment")
    parser.add_argument("--create-run", action="store_true")
    parser.add_argument("--run-params-json", default="{}", help="JSON object for pipeline params")
    parser.add_argument("--kfp-host", default=os.getenv("KFP_HOST", "http://ml-pipeline.kubeflow.svc.cluster.local:8888"))
    return parser.parse_args()


def import_pipeline_func(path: str):
    mod_name, func_name = path.split(":", 1)
    mod = __import__(mod_name, fromlist=[func_name])
    return getattr(mod, func_name)


def get_or_create_experiment(client: kfp.Client, name: str, namespace: str):
    try:
        return client.get_experiment(experiment_name=name, namespace=namespace)
    except Exception:
        return client.create_experiment(name=name, namespace=namespace)


def main() -> None:
    args = parse_args()
    token = read_token()

    pipeline_func = import_pipeline_func(args.pipeline_func)
    pipeline_yaml = Path(args.pipeline_yaml)
    pipeline_yaml.parent.mkdir(parents=True, exist_ok=True)

    compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=str(pipeline_yaml))
    print(f"[INFO] Compiled pipeline: {pipeline_yaml}")

    client = kfp.Client(host=args.kfp_host, namespace=args.namespace, existing_token=token)

    timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    pipeline_id = None
    try:
        pipeline_id = client.get_pipeline_id(args.pipeline_name)
    except Exception:
        pipeline_id = None

    if pipeline_id is None:
        created = client.upload_pipeline(
            pipeline_package_path=str(pipeline_yaml),
            pipeline_name=args.pipeline_name,
            description=args.description,
        )
        pipeline_id = getattr(created, "pipeline_id", None) or created.to_dict().get("pipeline", {}).get("id")
        print(f"[INFO] Created pipeline: {args.pipeline_name} (id={pipeline_id})")
    else:
        version_name = f"{args.pipeline_name}-{timestamp}"
        client.upload_pipeline_version(
            pipeline_package_path=str(pipeline_yaml),
            pipeline_version_name=version_name,
            pipeline_id=pipeline_id,
            description=args.description,
        )
        print(f"[INFO] Uploaded pipeline version: {version_name}")

    if not args.create_run:
        print("[INFO] Skipping run creation (--create-run not set)")
        return

    params = json.loads(args.run_params_json)
    exp = get_or_create_experiment(client, args.experiment_name, args.namespace)
    exp_id = exp.to_dict().get("id") or exp.to_dict().get("experiment_id")
    run_name = f"{args.pipeline_name}-run-{timestamp}"

    client.run_pipeline(
        experiment_id=exp_id,
        job_name=run_name,
        pipeline_package_path=str(pipeline_yaml),
        params=params,
    )
    print(f"[INFO] Submitted run: {run_name}")


if __name__ == "__main__":
    main()
