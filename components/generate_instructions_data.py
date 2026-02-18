from kfp import dsl


@dsl.component(base_image="python:3.11", packages_to_install=[])
def prepare_instruction_data(
    csv_path: str,
    output_dir: str,
    target_rows: int,
    val_ratio: float,
    seed: int,
    train_jsonl_out: dsl.OutputPath(str),
    val_jsonl_out: dsl.OutputPath(str),
    metadata_out: dsl.OutputPath(str),
):
    import csv
    import json
    import random
    import re
    from pathlib import Path

    unknown_markers = {"", "UNKNOWN", "N/A", "NA", "NULL", "NONE", "UNASSIGNED", "Unknown", "unknown"}
    strict_unknown_text = "Not available in provided data."

    def normalize(v):
        if v is None:
            return None
        s = str(v).strip()
        return None if s in unknown_markers else s

    def yes_no(v):
        x = normalize(v)
        if not x:
            return None
        x = x.upper()
        if x == "TRUE":
            return "Yes"
        if x == "FALSE":
            return "No"
        return None

    def title_ok(t):
        if not t:
            return False
        t = t.strip()
        if t.upper() in {"UNKNOWN", "N/A", "UNASSIGNED"}:
            return False
        if len(t) < 5:
            return False
        if t.count("(") != t.count(")"):
            return False
        if re.search(r"[^\w\s\-\&\.\,\(\)\/\+\'\:]", t):
            return False
        return True

    def product_name(row):
        t = row.get("TITLE", "")
        if title_ok(t):
            return t.strip()
        code = (row.get("PRODUCT_CODE") or "").strip()
        return f"product code {code}" if code else "this product"

    def safe(row, key):
        return normalize(row.get(key)) or strict_unknown_text

    def write_jsonl(path: Path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in rows:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")

    out_dir = Path(output_dir)
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    common_meta = (
        "Answer using only context. If requested info is UNKNOWN/blank/N/A/UNASSIGNED, "
        "reply exactly: 'Not available in provided data.'"
    )

    templates = [
        ("is_pepsi", ["TITLE", "PRODUCT_CODE", "IS_PEPSI_PRODUCT", "REPORTING_UPC", "TRADEMARK"],
         lambda r: f"Is {product_name(r)} a Pepsi product?",
         lambda r: f"{yes_no(r.get('IS_PEPSI_PRODUCT'))}." if yes_no(r.get("IS_PEPSI_PRODUCT")) else strict_unknown_text),
        ("reporting_upc", ["TITLE", "PRODUCT_CODE", "PRODUCT_CODE_TYPE", "REPORTING_UPC"],
         lambda r: f"What is the reporting UPC for {product_name(r)}?",
         lambda r: f"The reporting UPC is {safe(r, 'REPORTING_UPC')}." if normalize(r.get("REPORTING_UPC")) else strict_unknown_text),
        ("parent_company", ["TITLE", "PRODUCT_CODE", "TRADEMARK", "PARENT_COMPANY"],
         lambda r: f"Who is the parent company for {product_name(r)}?",
         lambda r: f"The parent company is {safe(r, 'PARENT_COMPANY')}." if normalize(r.get("PARENT_COMPANY")) else strict_unknown_text),
        ("manufacturer", ["TITLE", "PRODUCT_CODE", "TRADEMARK", "MANUFACTURER"],
         lambda r: f"Who is the manufacturer for {product_name(r)}?",
         lambda r: f"The manufacturer is {safe(r, 'MANUFACTURER')}." if normalize(r.get("MANUFACTURER")) else strict_unknown_text),
        ("category", ["TITLE", "IRI_CATEGORY_NAME", "IRI_SUB_CATEGORY_NAME"],
         lambda r: f"What category is {product_name(r)} in?",
         lambda r: f"It is in {safe(r, 'IRI_CATEGORY_NAME')} (sub-category: {safe(r, 'IRI_SUB_CATEGORY_NAME')})."
                   if normalize(r.get("IRI_CATEGORY_NAME")) and normalize(r.get("IRI_SUB_CATEGORY_NAME")) else strict_unknown_text),
        ("taxonomy", ["TITLE", "TAXONOMY_CATEGORY_BLENDED", "TAXONOMY_SUB_CATEGORY_BLENDED"],
         lambda r: f"What taxonomy category does {product_name(r)} belong to?",
         lambda r: f"Taxonomy category: {safe(r, 'TAXONOMY_CATEGORY_BLENDED')}; taxonomy sub-category: {safe(r, 'TAXONOMY_SUB_CATEGORY_BLENDED')}."
                   if normalize(r.get("TAXONOMY_CATEGORY_BLENDED")) and normalize(r.get("TAXONOMY_SUB_CATEGORY_BLENDED")) else strict_unknown_text),
        ("brand", ["TITLE", "FINANCE_BRAND_NAME", "SALES_BRAND_NAME"],
         lambda r: f"What are the finance and sales brands for {product_name(r)}?",
         lambda r: f"Finance brand: {safe(r, 'FINANCE_BRAND_NAME')}; Sales brand: {safe(r, 'SALES_BRAND_NAME')}."
                   if normalize(r.get("FINANCE_BRAND_NAME")) and normalize(r.get("SALES_BRAND_NAME")) else strict_unknown_text),
        ("flavor", ["TITLE", "STIBO_SUB_BRAND", "STIBO_FLAVOR", "STIBO_SUGAR_TYPE"],
         lambda r: f"What flavor is listed for {product_name(r)}?",
         lambda r: f"The listed flavor is {safe(r, 'STIBO_FLAVOR')}." if normalize(r.get("STIBO_FLAVOR")) else strict_unknown_text),
        ("active", ["TITLE", "IS_ACTIVE_PRODUCT"],
         lambda r: f"Is {product_name(r)} an active product?",
         lambda r: f"{yes_no(r.get('IS_ACTIVE_PRODUCT'))}." if yes_no(r.get("IS_ACTIVE_PRODUCT")) else strict_unknown_text),
        ("variety", ["TITLE", "IS_VARIETY_PACK"],
         lambda r: f"Is {product_name(r)} a variety pack?",
         lambda r: f"{yes_no(r.get('IS_VARIETY_PACK'))}." if yes_no(r.get("IS_VARIETY_PACK")) else strict_unknown_text),
        ("flamin_hot", ["TITLE", "IS_FLAMIN_HOT"],
         lambda r: f"Is {product_name(r)} marked as Flamin Hot?",
         lambda r: f"{yes_no(r.get('IS_FLAMIN_HOT'))}." if yes_no(r.get("IS_FLAMIN_HOT")) else strict_unknown_text),
    ]

    base_examples = []
    for row in rows:
        src_code = (row.get("PRODUCT_CODE") or "").strip()
        for template_id, context_fields, qf, af in templates:
            context = {k: (row.get(k, "").strip() if row.get(k) is not None else "") for k in context_fields}
            base_examples.append(
                {
                    "instruction": qf(row),
                    "context": context,
                    "answer": af(row),
                    "meta_instruction": common_meta,
                    "meta": {"template_id": template_id, "source_product_code": src_code},
                }
            )

    rng = random.Random(seed)
    if len(base_examples) >= target_rows:
        dataset = rng.sample(base_examples, target_rows)
    else:
        dataset = [base_examples[rng.randrange(0, len(base_examples))] for _ in range(target_rows)]
    rng.shuffle(dataset)

    split_idx = int(len(dataset) * (1 - val_ratio))
    train_rows = dataset[:split_idx]
    val_rows = dataset[split_idx:]

    train_path = out_dir / "finetune_train.jsonl"
    val_path = out_dir / "finetune_val.jsonl"
    write_jsonl(out_dir / "qa_base_all.jsonl", base_examples)
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    metadata = {
        "csv_rows": len(rows),
        "base_examples": len(base_examples),
        "target_rows": len(dataset),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "output_dir": str(out_dir),
    }

    Path(train_jsonl_out).write_text(str(train_path), encoding="utf-8")
    Path(val_jsonl_out).write_text(str(val_path), encoding="utf-8")
    Path(metadata_out).write_text(json.dumps(metadata), encoding="utf-8")
