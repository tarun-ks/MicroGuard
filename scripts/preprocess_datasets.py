"""
Step 3: Convert all datasets to unified format and create stratified splits.

Unified format:
{
    "query": str,
    "context": str,
    "answer": str,
    "label": "faithful" | "unfaithful",
    "source": str  # dataset provenance for stratification
}

Splits: 80% train, 10% validation, 10% test
Stratified by source AND label. Document-level splitting (no context leakage).
"""

import os
import json
import hashlib
from collections import Counter, defaultdict

import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# RAGBench conversion
# ─────────────────────────────────────────────────────────────────────────────

RAGBENCH_SUBSETS = [
    "covidqa", "cuad", "delucionqa", "emanual", "expertqa",
    "finqa", "hagrid", "hotpotqa", "msmarco", "pubmedqa",
    "tatqa", "techqa"
]


def convert_ragbench():
    """Convert RAGBench to unified format. Uses existing splits."""
    records = {"train": [], "validation": [], "test": []}

    for subset in RAGBENCH_SUBSETS:
        path = os.path.join(RAW_DIR, "ragbench", subset)
        if not os.path.exists(path):
            print(f"  Skipping {subset} (not found)")
            continue

        ds = load_from_disk(path)
        for split in ds:
            target_split = split  # keep original splits
            for ex in ds[split]:
                # Join documents list into single context string
                if isinstance(ex["documents"], list):
                    context = "\n\n".join(ex["documents"])
                else:
                    context = str(ex["documents"])

                # adherence_score is boolean: True=faithful, False=unfaithful
                label = "faithful" if ex["adherence_score"] else "unfaithful"

                records[target_split].append({
                    "query": ex["question"],
                    "context": context,
                    "answer": ex["response"],
                    "label": label,
                    "source": f"ragbench_{subset}",
                })

    for split, recs in records.items():
        labels = [r["label"] for r in recs]
        print(f"  RAGBench {split}: {len(recs)} examples — {Counter(labels)}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# RAGTruth conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_ragtruth():
    """Convert RAGTruth to unified format.
    Has train/test. We split 10% of train for validation.
    """
    ds = load_from_disk(os.path.join(RAW_DIR, "ragtruth"))
    records = {"train": [], "validation": [], "test": []}

    for split in ds:
        for ex in ds[split]:
            # Parse hallucination labels
            labels_proc = ex["hallucination_labels_processed"]
            if isinstance(labels_proc, str):
                labels_proc = json.loads(labels_proc)

            has_hallucination = (
                labels_proc.get("evident_conflict", 0) > 0 or
                labels_proc.get("baseless_info", 0) > 0
            )
            label = "unfaithful" if has_hallucination else "faithful"

            # Skip examples with no query (some summarization tasks)
            query = ex.get("query") or ""
            context = ex.get("context") or ""
            answer = ex.get("output") or ""

            if not context or not answer:
                continue

            records["test" if split == "test" else "train"].append({
                "query": query,
                "context": context,
                "answer": answer,
                "label": label,
                "source": f"ragtruth_{ex.get('task_type', 'unknown')}",
            })

    # Split some train into validation
    train_recs = records["train"]
    if len(train_recs) > 100:
        labels = [r["label"] for r in train_recs]
        train_idx, val_idx = train_test_split(
            range(len(train_recs)), test_size=0.1,
            stratify=labels, random_state=SEED
        )
        records["validation"] = [train_recs[i] for i in val_idx]
        records["train"] = [train_recs[i] for i in train_idx]

    for split, recs in records.items():
        labels = [r["label"] for r in recs]
        print(f"  RAGTruth {split}: {len(recs)} examples — {Counter(labels)}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# HaluBench conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_halubench():
    """Convert HaluBench to unified format.
    Has only test split. We split into 80/10/10.
    """
    ds = load_from_disk(os.path.join(RAW_DIR, "halubench"))
    all_records = []

    for ex in ds["test"]:
        label = "faithful" if ex["label"] == "PASS" else "unfaithful"

        all_records.append({
            "query": ex.get("question") or "",
            "context": ex.get("passage") or "",
            "answer": str(ex.get("answer") or ""),
            "label": label,
            "source": f"halubench_{ex.get('source_ds', 'unknown')}",
        })

    # Filter out empty examples
    all_records = [r for r in all_records if r["context"] and r["answer"]]

    # Stratified 80/10/10 split
    labels = [r["label"] for r in all_records]
    train_idx, temp_idx = train_test_split(
        range(len(all_records)), test_size=0.2,
        stratify=labels, random_state=SEED
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=temp_labels, random_state=SEED
    )

    records = {
        "train": [all_records[i] for i in train_idx],
        "validation": [all_records[i] for i in val_idx],
        "test": [all_records[i] for i in test_idx],
    }

    for split, recs in records.items():
        lbls = [r["label"] for r in recs]
        print(f"  HaluBench {split}: {len(recs)} examples — {Counter(lbls)}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Combine and save
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_by_context(records):
    """Remove exact duplicate context+answer pairs to avoid data leakage."""
    seen = set()
    unique = []
    dups = 0
    for r in records:
        key = hashlib.md5((r["context"][:500] + r["answer"][:500]).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(r)
        else:
            dups += 1
    if dups > 0:
        print(f"    Removed {dups} duplicate context+answer pairs")
    return unique


def main():
    print("Converting RAGBench...")
    ragbench = convert_ragbench()

    print("\nConverting RAGTruth...")
    ragtruth = convert_ragtruth()

    print("\nConverting HaluBench...")
    halubench = convert_halubench()

    # Combine all datasets per split
    combined = {}
    for split in ["train", "validation", "test"]:
        combined[split] = (
            ragbench.get(split, []) +
            ragtruth.get(split, []) +
            halubench.get(split, [])
        )
        combined[split] = deduplicate_by_context(combined[split])

    print("\n" + "=" * 70)
    print("Combined dataset summary:")
    print("=" * 70)
    total = 0
    for split in ["train", "validation", "test"]:
        recs = combined[split]
        labels = Counter(r["label"] for r in recs)
        sources = Counter(r["source"].split("_")[0] for r in recs)
        print(f"  {split}: {len(recs)} examples")
        print(f"    Labels: {dict(labels)}")
        print(f"    Sources: {dict(sources)}")
        total += len(recs)
    print(f"  Total: {total}")

    # Save as HuggingFace Dataset
    ds_dict = {}
    for split in ["train", "validation", "test"]:
        ds_dict[split] = Dataset.from_list(combined[split])
    ds = DatasetDict(ds_dict)

    save_path = os.path.join(PROCESSED_DIR, "microguard_combined")
    ds.save_to_disk(save_path)
    print(f"\nSaved combined dataset to {save_path}")

    # Also save per-source datasets for ablation studies (Step 9)
    for source_prefix in ["ragbench", "ragtruth", "halubench"]:
        per_source = {}
        for split in ["train", "validation", "test"]:
            recs = [r for r in combined[split] if r["source"].startswith(source_prefix)]
            if recs:
                per_source[split] = Dataset.from_list(recs)
        if per_source:
            source_ds = DatasetDict(per_source)
            source_path = os.path.join(PROCESSED_DIR, f"microguard_{source_prefix}")
            source_ds.save_to_disk(source_path)
            print(f"Saved {source_prefix} subset to {source_path}")

    # Save stats as JSON for paper
    stats = {
        "total_examples": total,
        "splits": {},
        "per_source": {},
    }
    for split in ["train", "validation", "test"]:
        labels = Counter(r["label"] for r in combined[split])
        stats["splits"][split] = {
            "total": len(combined[split]),
            "faithful": labels.get("faithful", 0),
            "unfaithful": labels.get("unfaithful", 0),
        }
    for r in combined["train"] + combined["validation"] + combined["test"]:
        src = r["source"].split("_")[0]
        stats["per_source"].setdefault(src, 0)
        stats["per_source"][src] += 1

    stats_path = os.path.join(PROCESSED_DIR, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
