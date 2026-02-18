from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["requests"],
)
def send_drift_alert(
    drift_result_json: str,
    webhook_url: str = "",
) -> str:
    import json
    import requests

    result = json.loads(drift_result_json)
    if result.get("drift_detected") != "true":
        return "no_alert"

    msg = {
        "text": f"[DRIFT] qwen-csv-qa drift detected: {json.dumps(result.get('details', {}), ensure_ascii=True)}"
    }

    if webhook_url.strip():
        try:
            requests.post(webhook_url, json=msg, timeout=10)
            return "alert_sent"
        except Exception as e:
            return f"alert_failed:{str(e)}"

    return "drift_detected_no_webhook"
