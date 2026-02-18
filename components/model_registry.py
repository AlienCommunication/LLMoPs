from kfp import dsl


@dsl.component(base_image="python:3.11", packages_to_install=["model-registry"])
def register_model_version(
    registry_server: str,
    model_name: str,
    model_version: str,
    storage_uri: str,
    model_id: str,
    description: str,
    registration_out: dsl.OutputPath(str),
):
    import inspect
    import json
    from pathlib import Path
    from urllib.parse import urlparse

    from model_registry import ModelRegistry

    def adapt_call(fn, args, kwargs):
        sig = inspect.signature(fn)
        bound_args = []

        pos_only = [
            p.name
            for p in sig.parameters.values()
            if p.kind == inspect.Parameter.POSITIONAL_ONLY and p.name != "self"
        ]
        for i, n in enumerate(pos_only):
            if i < len(args):
                bound_args.append(args[i])
            elif n in kwargs:
                bound_args.append(kwargs[n])

        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in sig.parameters and k not in pos_only and k != "self"
        }

        return fn(*bound_args, **filtered_kwargs)

    def get_attr_or_key(obj, *names):
        if obj is None:
            return None
        if isinstance(obj, dict):
            for n in names:
                if n in obj and obj[n] is not None:
                    return obj[n]
            return None
        for n in names:
            v = getattr(obj, n, None)
            if v is not None:
                return v
        return None

    raw_server = (registry_server or "").strip()
    if not raw_server:
        raise ValueError("registry_server is empty")

    parsed = urlparse(raw_server if "://" in raw_server else f"http://{raw_server}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"Cannot parse host from registry_server='{raw_server}'")
    port = parsed.port or 8080
    scheme = parsed.scheme or "http"
    is_secure = scheme == "https"
    server_url = f"{scheme}://{host}:{port}"

    init_sig = inspect.signature(ModelRegistry.__init__)

    ctor_candidates = [
        {
            "server_address": server_url,
            "author": "mlops-pipeline",
            "is_secure": is_secure,
        },
        {
            "server_address": server_url,
            "port": port,
            "author": "mlops-pipeline",
            "is_secure": is_secure,
        },
        {
            "server_address": host,
            "port": port,
            "author": "mlops-pipeline",
            "is_secure": is_secure,
        },
    ]

    registry = None
    last_ctor_err = None
    for candidate in ctor_candidates:
        try:
            init_kwargs = {k: v for k, v in candidate.items() if k in init_sig.parameters}
            registry = ModelRegistry(**init_kwargs)
            break
        except Exception as e:
            last_ctor_err = e

    if registry is None:
        raise RuntimeError(f"Unable to initialize ModelRegistry client: {last_ctor_err}")

    existing_model = None
    existing_version = None

    try:
        existing_model = adapt_call(registry.get_registered_model, [model_name], {"name": model_name})
    except Exception:
        existing_model = None

    try:
        existing_version = adapt_call(
            registry.get_model_version,
            [model_name, model_version],
            {"name": model_name, "version": model_version},
        )
    except Exception:
        existing_version = None

    created = False
    if existing_version is None:
        created = True
        rm = adapt_call(
            registry.register_model,
            [model_name, storage_uri],
            {
                "name": model_name,
                "uri": storage_uri,
                "model_format_name": "huggingface",
                "model_format_version": "1",
                "version": model_version,
                "description": description,
                "metadata": {"base_model_id": model_id, "storage_uri": storage_uri},
            },
        )

        try:
            existing_model = adapt_call(registry.get_registered_model, [model_name], {"name": model_name})
        except Exception:
            if existing_model is None:
                existing_model = rm

        try:
            existing_version = adapt_call(
                registry.get_model_version,
                [model_name, model_version],
                {"name": model_name, "version": model_version},
            )
        except Exception:
            if existing_version is None:
                existing_version = rm

    registered_model_id = get_attr_or_key(existing_model, "id", "registered_model_id")
    registered_version_id = get_attr_or_key(existing_version, "id", "model_version_id")

    payload = {
        "registry_server": raw_server,
        "registry_host": host,
        "registry_port": port,
        "registry_url": server_url,
        "registered_model_name": model_name,
        "registered_model_id": registered_model_id,
        "registered_version_name": model_version,
        "registered_version_id": registered_version_id,
        "storage_uri": storage_uri,
        "base_model_id": model_id,
        "created": created,
    }

    print("MODEL_REGISTRY_RESULT=" + json.dumps(payload, ensure_ascii=True))
    Path(registration_out).write_text(json.dumps(payload), encoding="utf-8")

    if not registered_model_id:
        raise RuntimeError("Model registry registration failed: registered_model_id missing")
    if not registered_version_id:
        raise RuntimeError("Model registry registration failed: registered_version_id missing")
