"""
extract_responses.py — Extract model responses from Inspect log files
======================================================================
Reads the Zstd-compressed .eval log files, extracts the assistant's
response text for each sample, and merges with results.csv to create
results_with_responses.csv.

This is needed for the judge variation study (Part 2), which re-scores
existing responses with different judge configurations.
"""

import zipfile
import json
import struct
import os
from pathlib import Path

import zstandard
import pandas as pd


def read_zstd_zip_entry(filepath, entry_name):
    """Read a single entry from a Zstd-compressed zip file."""
    with open(filepath, "rb") as f:
        zf = zipfile.ZipFile(f)
        info = zf.getinfo(entry_name)
        f.seek(info.header_offset)
        fheader = f.read(30)
        fname_len = struct.unpack("<H", fheader[26:28])[0]
        extra_len = struct.unpack("<H", fheader[28:30])[0]
        f.read(fname_len + extra_len)
        compressed = f.read(info.compress_size)
        dctx = zstandard.ZstdDecompressor()
        raw = dctx.decompress(compressed, max_output_size=info.file_size)
        return json.loads(raw)


def list_sample_entries(filepath):
    """List all sample entry names in a log file."""
    with open(filepath, "rb") as f:
        zf = zipfile.ZipFile(f)
        return [n for n in zf.namelist() if n.startswith("samples/")]


def extract_response_text(sample_data):
    """Extract the assistant's response text from a sample."""
    messages = sample_data.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("source") == "generate":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Content is a list of blocks
                texts = [
                    block.get("text", "")
                    for block in content
                    if block.get("type") == "text"
                ]
                return " ".join(texts)
            elif isinstance(content, str):
                return content
    return ""


def extract_log_metadata(filepath):
    """Extract model name and run metadata from the log header."""
    try:
        header = read_zstd_zip_entry(filepath, "header.json")
        model = header.get("model", "")
        # Try to get run_id and scaffold from eval config or metadata
        return header
    except Exception:
        return {}


def extract_all_responses(logs_dir="logs", results_csv="results/results.csv"):
    """Extract responses from all log files and merge with results."""

    logs_path = Path(logs_dir)
    log_files = sorted(logs_path.glob("*.eval"))
    print(f"Found {len(log_files)} log files in {logs_dir}/")

    # Load existing results
    results = pd.read_csv(results_csv)
    print(f"Loaded {len(results)} rows from {results_csv}")
    print(f"Columns: {list(results.columns)}")

    # We need to match log samples to results rows.
    # Strategy: extract all samples from all logs, then match on
    # (model, sample_id, question) since run_id/scaffold aren't in the log filename.

    all_responses = []
    for i, log_file in enumerate(log_files):
        if i % 10 == 0:
            print(f"  Processing log {i+1}/{len(log_files)}: {log_file.name}")

        try:
            # Get header for model info
            header = read_zstd_zip_entry(log_file, "header.json")
            model_name = header.get("eval", {}).get("model", "")
            # Clean model name
            model_name = model_name.replace("anthropic/", "")
            # Remove version suffix if present (e.g., claude-haiku-4-5-20251001 → claude-haiku-4-5)
            for known_model in ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-6"]:
                if model_name.startswith(known_model):
                    model_name = known_model
                    break

            # Get sample entries
            sample_entries = list_sample_entries(log_file)

            for entry_name in sample_entries:
                try:
                    sample = read_zstd_zip_entry(log_file, entry_name)
                    sample_id = sample.get("id")
                    question = sample.get("input", "")
                    response_text = extract_response_text(sample)

                    all_responses.append({
                        "log_file": log_file.name,
                        "model": model_name,
                        "sample_id": sample_id,
                        "question": question,
                        "response": response_text,
                    })
                except Exception as e:
                    continue

        except Exception as e:
            print(f"  WARNING: Could not process {log_file.name}: {e}")
            continue

    print(f"\nExtracted {len(all_responses)} responses from log files")

    # Convert to DataFrame
    resp_df = pd.DataFrame(all_responses)

    # Check for duplicates (multiple logs may have same model × sample_id)
    print(f"Unique (model, sample_id) pairs: {resp_df.groupby(['model', 'sample_id']).ngroups}")

    # Merge with results on model + sample_id + question
    # Since multiple runs have the same model + sample_id, we need to handle
    # the many-to-many carefully. Each results row needs ONE response.
    # For runs with the same model/scaffold/sample_id, responses differ by seed.
    # The log filename contains the timestamp, which maps to run order.

    # Sort responses by log file (chronological) to align with results
    resp_df = resp_df.sort_values(["model", "sample_id", "log_file"])

    # For each (model, sample_id), assign a sequence number
    resp_df["seq"] = resp_df.groupby(["model", "sample_id"]).cumcount()

    # Similarly, for results, assign sequence within (model, sample_id)
    results = results.sort_values(["model", "sample_id", "run_id", "scaffold"])
    results["seq"] = results.groupby(["model", "sample_id"]).cumcount()

    # Merge on model + sample_id + seq
    merged = results.merge(
        resp_df[["model", "sample_id", "seq", "response"]],
        on=["model", "sample_id", "seq"],
        how="left",
    )

    n_matched = merged["response"].notna().sum()
    n_missing = merged["response"].isna().sum()
    print(f"\nMerge results:")
    print(f"  Matched: {n_matched}")
    print(f"  Missing: {n_missing}")

    # Drop the seq column
    merged = merged.drop(columns=["seq"])

    # Save
    output_file = results_csv.replace(".csv", "_with_responses.csv")
    merged.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")
    print(f"  Columns: {list(merged.columns)}")

    # Verify a few samples
    print(f"\nSample responses (first 3):")
    for _, row in merged.head(3).iterrows():
        print(f"  Model: {row['model']}, Q: {row['question'][:50]}...")
        resp = row.get("response", "")
        if pd.notna(resp):
            print(f"  R: {str(resp)[:100]}...")
        else:
            print(f"  R: MISSING")
        print()

    return merged


if __name__ == "__main__":
    extract_all_responses()