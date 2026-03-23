"""
Step 2: Download all datasets for MicroGuard experiments.
Saves raw datasets to data/raw/
"""

import os
import json
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)


def download_ragbench():
    """Download RAGBench — 100K examples with TRACe labels.
    RAGBench has 12 sub-datasets, each with train/val/test splits.
    """
    print("=" * 60)
    print("Downloading RAGBench (galileo-ai/ragbench)...")
    subsets = [
        "covidqa", "cuad", "delucionqa", "emanual", "expertqa",
        "finqa", "hagrid", "hotpotqa", "msmarco", "pubmedqa",
        "tatqa", "techqa"
    ]
    save_path = os.path.join(DATA_DIR, "ragbench")
    os.makedirs(save_path, exist_ok=True)
    total = 0
    try:
        for subset in subsets:
            print(f"  Loading subset: {subset}...")
            ds = load_dataset("galileo-ai/ragbench", subset)
            subset_path = os.path.join(save_path, subset)
            ds.save_to_disk(subset_path)
            for split in ds:
                n = len(ds[split])
                total += n
                print(f"    {split}: {n} examples")
            if len(ds) > 0:
                first_split = list(ds.keys())[0]
                print(f"    Columns: {ds[first_split].column_names}")
        print(f"Total RAGBench examples: {total}")
        print(f"Saved to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download RAGBench: {e}")
        return False


def download_ragtruth():
    """Download RAGTruth — real LLM outputs with hallucination annotations."""
    print("=" * 60)
    print("Downloading RAGTruth...")
    try:
        ds = load_dataset("wandb/RAGTruth-processed")
        save_path = os.path.join(DATA_DIR, "ragtruth")
        ds.save_to_disk(save_path)
        for split in ds:
            print(f"  {split}: {len(ds[split])} examples")
            if len(ds[split]) > 0:
                print(f"  Columns: {ds[split].column_names}")
        print(f"Saved to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download RAGTruth: {e}")
        return False


def download_halubench():
    """Download HaluBench — hallucination detection benchmark."""
    print("=" * 60)
    print("Downloading HaluBench...")
    try:
        ds = load_dataset("PatronusAI/HaluBench")
        save_path = os.path.join(DATA_DIR, "halubench")
        ds.save_to_disk(save_path)
        for split in ds:
            print(f"  {split}: {len(ds[split])} examples")
            if len(ds[split]) > 0:
                print(f"  Columns: {ds[split].column_names}")
        print(f"Saved to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download HaluBench: {e}")
        return False


def download_fallbacks():
    """Download fallback datasets in case primary ones fail."""
    fallbacks = {}

    print("=" * 60)
    print("Downloading fallback: TruthfulQA...")
    try:
        ds = load_dataset("truthful_qa", "multiple_choice")
        save_path = os.path.join(DATA_DIR, "truthful_qa")
        ds.save_to_disk(save_path)
        for split in ds:
            print(f"  {split}: {len(ds[split])} examples")
        fallbacks["truthful_qa"] = True
    except Exception as e:
        print(f"  Failed: {e}")
        fallbacks["truthful_qa"] = False

    print("=" * 60)
    print("Downloading fallback: HaluEval...")
    try:
        ds = load_dataset("pminervini/HaluEval", "qa_samples")
        save_path = os.path.join(DATA_DIR, "halueval")
        ds.save_to_disk(save_path)
        for split in ds:
            print(f"  {split}: {len(ds[split])} examples")
        fallbacks["halueval"] = True
    except Exception as e:
        print(f"  Failed: {e}")
        fallbacks["halueval"] = False

    return fallbacks


if __name__ == "__main__":
    results = {}

    results["ragbench"] = download_ragbench()
    results["ragtruth"] = download_ragtruth()
    results["halubench"] = download_halubench()

    # Check if we need fallbacks
    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n{'=' * 60}")
        print(f"Primary datasets that failed: {failed}")
        print("Downloading fallback datasets...")
        fallback_results = download_fallbacks()
        results.update({f"fallback_{k}": v for k, v in fallback_results.items()})

    # Save download report
    report_path = os.path.join(DATA_DIR, "download_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Download Summary:")
    for name, status in results.items():
        print(f"  {name}: {'OK' if status else 'FAILED'}")
    print(f"\nReport saved to {report_path}")
