from kfp import dsl


@dsl.component(base_image="python:3.11", packages_to_install=["kubernetes", "PyYAML"])
def deploy_inferenceservice(
    model_name: str,
    namespace: str,
    storage_uri: str,
    runtime_name: str,
    cpu_request: str,
    cpu_limit: str,
    mem_request: str,
    mem_limit: str,
    manifest_out: dsl.OutputPath(str),
    deploy_status_out: dsl.OutputPath(str),
):
    import json
    import time
    from pathlib import Path

    import yaml
    from kubernetes import client, config

    manifest = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {"name": model_name, "namespace": namespace},
        "spec": {
            "predictor": {
                "model": {
                    "runtime": runtime_name,
                    "modelFormat": {"name": "huggingface"},
                    "storageUri": storage_uri,
                    "resources": {
                        "requests": {"cpu": cpu_request, "memory": mem_request, "nvidia.com/gpu": "1"},
                        "limits": {"cpu": cpu_limit, "memory": mem_limit, "nvidia.com/gpu": "1"},
                    },
                },
                "nodeSelector": {
                    "agentpool": "gpuac3d",
                    "kubernetes.azure.com/accelerator": "nvidia",
                },
                "tolerations": [{"key": "type", "operator": "Equal", "value": "gpu", "effect": "NoSchedule"}],
            }
        },
    }

    Path(manifest_out).write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    config.load_incluster_config()
    api = client.CustomObjectsApi()
    group = "serving.kserve.io"
    version = "v1beta1"
    plural = "inferenceservices"

    try:
        api.get_namespaced_custom_object(group, version, namespace, plural, model_name)
        api.patch_namespaced_custom_object(group, version, namespace, plural, model_name, manifest)
    except client.exceptions.ApiException as e:
        if e.status == 404:
            api.create_namespaced_custom_object(group, version, namespace, plural, manifest)
        else:
            raise

    deadline = time.time() + 1200
    latest = None
    while time.time() < deadline:
        latest = api.get_namespaced_custom_object(group, version, namespace, plural, model_name)
        conditions = latest.get("status", {}).get("conditions", [])
        ready = [c for c in conditions if c.get("type") == "Ready"]
        if ready and ready[-1].get("status") == "True":
            Path(deploy_status_out).write_text(json.dumps(latest), encoding="utf-8")
            return
        time.sleep(10)

    raise RuntimeError(f"InferenceService {namespace}/{model_name} did not become Ready in time")
