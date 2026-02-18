from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "transformers==4.46.3",
        "trl==0.11.4",
        "peft==0.13.2",
        "accelerate==1.1.1",
        "datasets==3.1.0",
        "bitsandbytes",
        "sentencepiece",
        "rich",
    ],
)
def fine_tune_lora(
    train_jsonl: str,
    val_jsonl: str,
    adapter_output_dir: str,
    model_id: str,
    max_seq_len: int,
    batch_size: int,
    grad_acc: int,
    lr: float,
    epochs: int,
    metrics_out: dsl.OutputPath(str),
):
    import json
    import os
    from pathlib import Path

    import torch
    from datasets import Dataset
    from huggingface_hub import login
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)

    def load_jsonl(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    train_rows = load_jsonl(train_jsonl)
    val_rows = load_jsonl(val_jsonl)

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    system = (
        "Answer using only context. If requested info is UNKNOWN/blank/N/A/UNASSIGNED, "
        "reply exactly: 'Not available in provided data.'"
    )

    def to_text(ex):
        return {
            "text": (
                f"System: {system}\\n"
                f"User: Question: {ex['instruction']}\\n"
                f"Context: {json.dumps(ex['context'], ensure_ascii=False)}\\n"
                f"Assistant: {ex['answer']}"
            )
        }

    train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(to_text, remove_columns=val_ds.column_names)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_4bit = True
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[WARN] 4-bit loading failed; falling back to fp16 model load. Error: {e}")
        use_4bit = False
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    out = Path(adapter_output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(out / "workdir"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        logging_steps=20,
        eval_steps=100,
        save_steps=100,
        fp16=True,
        bf16=False,
        max_seq_length=max_seq_len,
        packing=False,
        dataset_text_field="text",
        gradient_checkpointing=True,
        report_to="none",
        optim=("paged_adamw_8bit" if use_4bit else "adamw_torch"),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        args=sft_config,
    )

    train_output = trainer.train()

    final_dir = out / "final"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = {
        "train_loss": float(train_output.training_loss),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "model_id": model_id,
        "adapter_path": str(final_dir),
    }
    Path(metrics_out).write_text(json.dumps(metrics), encoding="utf-8")
