from datasets import load_dataset
from inspect_ai.dataset import Sample
import random

def load_truthfulqa_samples(n=50, seed=42):
    """ TruthfulQA 'generation' split has open-ended questions with reference 
    answers
    """

    ds = load_dataset("truthful_qa", "generation", split="validation")

    # stratify by category
    by_category = {}

    for row in ds:
        by_category.setdefault(row['category'], []).append(row)
    
    rng = random.Random(42)

    categories = list(by_category.keys())

    samples = []

    per_cat = max(2, 50 // len(categories))
    for cat in categories:
        rows = by_category[cat]
        for row in rows[:per_cat]:
            samples.append(Sample(
                input=row["question"],
                target=row["best_answer"],
                metadata={
                    "category": row["category"],
                    "correct_answers": row["correct_answers"],
                    "incorrect_answers": row["incorrect_answers"],
                },
            ))
            if len(samples) >= n:
                break
        if len(samples) >= n:
            break
    
    return samples

if __name__ == "__main__":
    samples = load_truthfulqa_samples(n=50)
    print(f"Loaded {len(samples)} samples")
    print(f"\nExample question: {samples[0].input}")
    print(f"Reference best answer: {samples[0].target}")
    print(f"Category: {samples[0].metadata['category']}")
    print(f"# of correct answer variants: {len(samples[0].metadata['correct_answers'])}")
    print(f"# of incorrect answer variants: {len(samples[0].metadata['incorrect_answers'])}")