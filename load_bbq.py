"""
load_bbq.py — Load and sample from BBQ (Bias Benchmark for QA)
================================================================
Downloads BBQ from HuggingFace (heegyu/bbq), takes a stratified sample
across bias categories and context types (ambiguous vs disambiguated),
and formats as Inspect samples.

Each item becomes a multiple-choice question with 3 options.
Scoring is deterministic — no LLM judge needed.

BBQ fields:
  - context: the scenario text
  - question: the question about the scenario
  - ans0, ans1, ans2: three answer options
  - label: index of the correct answer (0, 1, or 2)
  - context_condition: "ambig" or "disambig"
  - category: bias category (Age, Gender_identity, Race_ethnicity, etc.)
  - question_polarity: "neg" or "nonneg"
  - target_loc: index of the stereotyped answer

Dependencies:
  pip install datasets --break-system-packages
"""

import json
import random
import urllib.request
from pathlib import Path
from inspect_ai.dataset import Sample


# BBQ categories (9 primary, excluding intersectional for simplicity)
PRIMARY_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Religion",
    "SES",
    "Sexual_orientation",
]

# GitHub raw URLs for BBQ data files
BBQ_BASE_URL = (
    "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data"
)

# Cache directory
BBQ_CACHE_DIR = Path(__file__).parent / ".bbq_cache"


def _download_category(category: str) -> list[dict]:
    """Download and cache a BBQ category JSONL file from GitHub."""
    BBQ_CACHE_DIR.mkdir(exist_ok=True)
    cache_path = BBQ_CACHE_DIR / f"{category}.jsonl"

    if not cache_path.exists():
        url = f"{BBQ_BASE_URL}/{category}.jsonl"
        print(f"  Downloading {category} from {url}...")
        try:
            urllib.request.urlretrieve(url, cache_path)
        except Exception as e:
            print(f"  WARNING: Could not download '{category}': {e}")
            return []

    items = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    return items


def load_bbq_samples(
    per_category: int = 30,
    categories: list[str] | None = None,
    context_conditions: list[str] | None = None,
    question_polarity: str | None = "neg",
    seed: int = 42,
) -> list[Sample]:
    """Load a stratified sample of BBQ items.

    Args:
        per_category: Number of items per category (split evenly between
                      ambiguous and disambiguated if both conditions used).
        categories: Which bias categories to include. Defaults to all 9 primary.
        context_conditions: ["ambig", "disambig"] or one of them. Default both.
        question_polarity: "neg" (negative stereotype questions only),
                          "nonneg", or None (both). Default "neg" for cleaner
                          bias measurement.
        seed: Random seed for sampling.

    Returns:
        List of Inspect Sample objects.
    """
    random.seed(seed)

    if categories is None:
        categories = PRIMARY_CATEGORIES
    if context_conditions is None:
        context_conditions = ["ambig", "disambig"]

    samples = []
    sample_id = 0

    for category in categories:
        all_items = _download_category(category)
        if not all_items:
            continue

        print(f"  {category}: {len(all_items)} total items")

        for condition in context_conditions:
            # Filter by context condition and polarity
            filtered = [
                item for item in all_items
                if item.get("context_condition") == condition
            ]

            if question_polarity:
                filtered = [
                    item for item in filtered
                    if item.get("question_polarity") == question_polarity
                ]

            if not filtered:
                print(f"  WARNING: No items for {category}/{condition}/"
                      f"{question_polarity}")
                continue

            # Sample
            n_sample = per_category // len(context_conditions)
            if n_sample > len(filtered):
                print(f"  WARNING: Only {len(filtered)} items available for "
                      f"{category}/{condition}, requested {n_sample}")
                n_sample = len(filtered)

            selected = random.sample(filtered, n_sample)

            for item in selected:
                sample_id += 1

                # Build the answer options
                answers = [item["ans0"], item["ans1"], item["ans2"]]
                correct_label = item["label"]
                correct_answer = answers[correct_label]

                # Determine if the correct answer is the stereotyped one
                target_loc = -1
                answer_info = item.get("answer_info", {})
                stereotyped_groups = item.get("additional_metadata", {}).get(
                    "stereotyped_groups", []
                )
                if answer_info and stereotyped_groups:
                    for ans_key in ["ans0", "ans1", "ans2"]:
                        info = answer_info.get(ans_key, [])
                        # info is like ['grandfather', 'old'] — second element is the group label
                        if len(info) >= 2 and info[1] in stereotyped_groups:
                            target_loc = int(ans_key[-1])  # 0, 1, or 2
                            break
                correct_is_stereotyped = (correct_label == target_loc)

                # Build the prompt: context + question + options
                prompt = (
                    f"{item['context']}\n\n"
                    f"{item['question']}\n\n"
                    f"(A) {answers[0]}\n"
                    f"(B) {answers[1]}\n"
                    f"(C) {answers[2]}\n\n"
                    f"Answer with just the letter: A, B, or C."
                )

                # Map label index to letter
                label_to_letter = {0: "A", 1: "B", 2: "C"}
                correct_letter = label_to_letter[correct_label]

                # Determine stereotyped answer letter
                stereotyped_letter = (
                    label_to_letter[target_loc] if target_loc in (0, 1, 2) else None
                )

                sample = Sample(
                    input=prompt,
                    target=correct_letter,
                    id=str(sample_id),
                    metadata={
                        "category": category,
                        "context_condition": condition,
                        "question_polarity": item.get("question_polarity", ""),
                        "correct_answer_text": correct_answer,
                        "correct_label": correct_label,
                        "target_loc": target_loc,
                        "correct_is_stereotyped": correct_is_stereotyped,
                        "stereotyped_letter": stereotyped_letter,
                        "question_text": item["question"],
                        "ans0": answers[0],
                        "ans1": answers[1],
                        "ans2": answers[2],
                    },
                )
                samples.append(sample)

    print(f"Loaded {len(samples)} BBQ samples across {len(categories)} categories")
    print(f"  Context conditions: {context_conditions}")
    print(f"  Question polarity: {question_polarity or 'both'}")
    print(f"  Per category: {per_category}")

    # Summary by category × condition
    from collections import Counter
    counts = Counter(
        (s.metadata["category"], s.metadata["context_condition"]) for s in samples
    )
    for (cat, cond), n in sorted(counts.items()):
        print(f"    {cat:<25} {cond:<10} {n:>4}")

    return samples


if __name__ == "__main__":
    # Test loading
    samples = load_bbq_samples(per_category=30)
    print(f"\nTotal samples: {len(samples)}")
    print(f"\nExample item:")
    s = samples[0]
    print(f"  ID: {s.id}")
    print(f"  Input: {s.input[:200]}...")
    print(f"  Target: {s.target}")
    print(f"  Category: {s.metadata['category']}")
    print(f"  Condition: {s.metadata['context_condition']}")
    print(f"  Stereotyped letter: {s.metadata['stereotyped_letter']}")