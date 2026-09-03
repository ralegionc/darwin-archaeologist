"""
interface/elicitor.py

Failure elicitation protocol runner.

Systematically runs all failure prompts, collects responses,
and produces a structured report. This is the research engine.

Usage:
    python interface/elicitor.py --prompts data/failure_prompts.json --output results/
    python interface/elicitor.py --category temporal_lock --runs 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, ELICITATION_RUNS_PER_PROMPT
from pipeline.model import DarwinModel


def run_elicitation(
    model: DarwinModel,
    prompts_path: Path,
    output_dir: Path,
    category_filter: str = None,
    runs_per_prompt: int = ELICITATION_RUNS_PER_PROMPT,
):
    """Run the full failure elicitation battery."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(prompts_path, encoding="utf-8") as f:
        prompt_data = json.load(f)

    categories = prompt_data["categories"]
    if category_filter:
        categories = {k: v for k, v in categories.items() if k == category_filter}

    all_results = []
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print(f"── Failure Elicitation Battery ───────────────────────")
    print(f"  Session: {session_id}")
    print(f"  Categories: {list(categories.keys())}")
    print(f"  Runs per prompt: {runs_per_prompt}")
    total_prompts = sum(len(cat["prompts"]) for cat in categories.values())
    print(f"  Total prompts: {total_prompts}")
    print(f"  Total API calls: {total_prompts * runs_per_prompt}")
    print()

    for cat_name, cat_data in categories.items():
        print(f"\n── {cat_name} ({'─' * (45 - len(cat_name))})")
        cat_results = []

        for prompt_spec in cat_data["prompts"]:
            prompt_id = prompt_spec["id"]
            prompt_text = prompt_spec["prompt"]
            date_context = prompt_spec.get("date_context", "not specified")
            filter_before_year = prompt_spec.get("filter_before_year")
            expected_failure = prompt_spec.get("expected_failure", "")
            ground_truth = prompt_spec.get("ground_truth", "")

            print(f"\n  [{prompt_id}] {prompt_text[:60]}...")
            print(f"  Date context: {date_context}")
            print(f"  Running {runs_per_prompt} times...")

            responses = []
            for run_i in range(runs_per_prompt):
                print(f"    Run {run_i + 1}/{runs_per_prompt}...", end=" ", flush=True)
                try:
                    response = model.query(
                        prompt=prompt_text,
                        date_context=date_context,
                        filter_before_year=filter_before_year,
                        failure_category=cat_name,
                    )
                    responses.append(response.to_dict())
                    print(f"✓ ({len(response.response_text)} chars, "
                          f"{len(response.retrieved_passages)} passages)")
                except Exception as e:
                    print(f"✗ {e}")
                    responses.append({"error": str(e)})

                time.sleep(0.5)  # rate limit buffer

            # Analyze variance across runs
            texts = [r.get("response_text", "") for r in responses if "response_text" in r]
            variance_note = analyze_variance(texts)

            result = {
                "session_id": session_id,
                "category": cat_name,
                "category_description": cat_data["description"],
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "date_context": date_context,
                "filter_before_year": filter_before_year,
                "expected_failure": expected_failure,
                "ground_truth": ground_truth,
                "runs": runs_per_prompt,
                "responses": responses,
                "variance_analysis": variance_note,
                "elicited_at": datetime.utcnow().isoformat(),
            }

            cat_results.append(result)
            all_results.append(result)

            # Save incrementally
            out_path = output_dir / f"{session_id}_{prompt_id}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        # Save category summary
        cat_path = output_dir / f"{session_id}_{cat_name}_summary.json"
        cat_path.write_text(json.dumps(cat_results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Full session report
    report = {
        "session_id": session_id,
        "completed_at": datetime.utcnow().isoformat(),
        "model_backend": model.backend,
        "total_prompts": len(all_results),
        "total_runs": sum(r["runs"] for r in all_results),
        "results": all_results,
    }
    report_path = output_dir / f"{session_id}_full_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n\n── Session complete ───────────────────────────────────")
    print(f"  Prompts run: {len(all_results)}")
    print(f"  Results → {output_dir}")
    print(f"  Full report: {report_path.name}")

    return report


def analyze_variance(texts: list[str]) -> dict:
    """
    Measure variance across multiple runs of the same prompt.

    High variance = model uncertainty (less dangerous — at least it knows it doesn't know).
    Low variance + wrong = confident confabulation (most dangerous failure mode).
    """
    if len(texts) < 2:
        return {"note": "insufficient runs for variance analysis"}

    # Rough lexical overlap as variance proxy
    word_sets = [set(t.lower().split()) for t in texts if t]
    if not word_sets:
        return {"note": "no valid responses"}

    # Average pairwise Jaccard similarity
    similarities = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            if union > 0:
                similarities.append(intersection / union)

    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    avg_length = sum(len(t) for t in texts) // len(texts)

    variance_level = (
        "low" if avg_similarity > 0.6 else
        "medium" if avg_similarity > 0.35 else
        "high"
    )

    return {
        "avg_lexical_similarity": round(avg_similarity, 3),
        "variance_level": variance_level,
        "avg_response_length": avg_length,
        "interpretation": {
            "low": "Responses are very similar — consistent but potentially consistently wrong",
            "medium": "Moderate variation — model has some uncertainty",
            "high": "High variation — model is genuinely uncertain about this domain",
        }[variance_level],
    }


def print_summary_report(output_dir: Path, session_id: str = None):
    """Print a human-readable summary of elicitation results."""
    if session_id:
        report_files = list(output_dir.glob(f"{session_id}_full_report.json"))
    else:
        report_files = sorted(output_dir.glob("*_full_report.json"))

    if not report_files:
        print("No reports found.")
        return

    report_path = report_files[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    print(f"\n{'='*60}")
    print(f"FAILURE ELICITATION REPORT")
    print(f"Session: {report['session_id']}")
    print(f"Model: {report['model_backend']}")
    print(f"{'='*60}")

    by_category = {}
    for result in report["results"]:
        cat = result["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(result)

    for cat, results in by_category.items():
        print(f"\n── {cat.upper()} ──")
        for r in results:
            variance = r.get("variance_analysis", {})
            print(f"\n  [{r['prompt_id']}] {r['prompt'][:55]}...")
            print(f"  Date context: {r['date_context']}")
            print(f"  Variance: {variance.get('variance_level', 'n/a')} "
                  f"(similarity: {variance.get('avg_lexical_similarity', 'n/a')})")
            print(f"  Expected failure: {r['expected_failure'][:80]}...")
            if r['responses']:
                first = r['responses'][0]
                if 'response_text' in first:
                    print(f"  Sample response: {first['response_text'][:150]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=Path("data/failure_prompts.json"))
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--category", help="Run only this category")
    parser.add_argument("--runs", type=int, default=ELICITATION_RUNS_PER_PROMPT)
    parser.add_argument("--backend", default=None, help="Override LLM backend")
    parser.add_argument("--summary", action="store_true", help="Print summary of existing results")
    args = parser.parse_args()

    if args.summary:
        print_summary_report(args.output)
        return

    model = DarwinModel(backend=args.backend) if args.backend else DarwinModel()
    run_elicitation(model, args.prompts, args.output, args.category, args.runs)


if __name__ == "__main__":
    main()
