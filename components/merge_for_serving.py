from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "transformers==4.46.3",
        "peft==0.13.2",
        "accelerate==1.1.1",
        "sentencepiece",
    ],
)
def merge_for_serving(
    base_model_id: str,
    adapter_dir: str,
    merged_output_dir: str,
    merged_path_out: dsl.OutputPath(str),
):
    import os
    from pathlib import Path

    import torch
    from huggingface_hub import login
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["USE_TF"] = "0"

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)

    out_dir = Path(merged_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    max_memory = None
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        usable = int(total * 0.85)
        max_memory = {0: usable, "cpu": "32GiB"}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        max_memory=max_memory,
    )

    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    Path(merged_path_out).write_text(str(out_dir), encoding="utf-8")
