"""
interface/annotator.py

Human validator annotation tool.

Presents model outputs alongside real Darwin passages for blind rating.
Validators rate authenticity 1-5 and identify specific failure points.
Their disagreements are as interesting as the model's errors.

Usage:
    python interface/annotator.py --results results/ --session SESSION_ID
    python interface/annotator.py --export results/ --output annotations.csv
"""

import argparse
import json
import csv
import sys
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, AUTHENTICITY_SCALE


RATING_GUIDE = """
Authenticity Rating Scale (1-5):
  5 = Indistinguishable from real Darwin — voice, knowledge, register all authentic
  4 = Mostly authentic with minor anachronisms or register slips
  3 = Recognizably Darwin-adjacent but with noticeable errors
  2 = Darwin-flavored but substantially wrong (wrong knowledge, wrong register)
  1 = Could not plausibly be Darwin — wrong era, wrong voice, obvious confabulation
"""

FAILURE_TAGS = [
    "temporal_lock",      # Uses knowledge Darwin couldn't have had
    "register_wrong",     # Wrong emotional register for context
    "confabulation",      # Invented plausible-sounding facts
    "too_formal",         # Over-indexes on published voice
    "too_casual",         # Under-indexes on Darwin's precision
    "anachronistic_concept",  # Concept not available in Darwin's time
    "correct",            # No failure detected
    "uncertain",          # Can't assess without more context
]


def load_results(results_dir: Path, session_id: str = None) -> list[dict]:
    if session_id:
        pattern = f"{session_id}_*.json"
    else:
        pattern = "*_full_report.json"

    files = sorted(results_dir.glob(pattern))
    if not files:
        print(f"No results found in {results_dir}")
        return []

    # Load from full report if available
    report_files = [f for f in files if "full_report" in f.name]
    if report_files:
        report = json.loads(report_files[-1].read_text(encoding="utf-8"))
        return report.get("results", [])

    # Otherwise load individual result files
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "prompt_id" in data:
                results.append(data)
        except Exception:
            pass
    return results


def annotate_result(result: dict, validator_name: str) -> dict:
    """Interactive annotation of a single result."""
    print(f"\n{'='*65}")
    print(f"PROMPT: {result['prompt']}")
    print(f"Date context: {result['date_context']}")
    print(f"Category: {result['category']}")
    print(f"{'='*65}")

    # Show one model response at a time (randomized order if multiple)
    responses = result.get("responses", [])
    valid_responses = [r for r in responses if "response_text" in r]
    if not valid_responses:
        print("  [No valid responses to annotate]")
        return None

    response = random.choice(valid_responses)

    print(f"\n[RESPONSE]\n{response['response_text']}")
    print()

    if response.get("passages"):
        print("[RETRIEVED SOURCES]")
        for p in response["passages"][:2]:
            print(f"  • {p['citation']} (score: {p['score']})")
            print(f"    \"{p['text'][:100]}...\"")
        print()

    print(RATING_GUIDE)

    # Get rating
    while True:
        try:
            rating = int(input(f"Authenticity rating (1-{AUTHENTICITY_SCALE}): ").strip())
            if 1 <= rating <= AUTHENTICITY_SCALE:
                break
            print(f"  Enter a number between 1 and {AUTHENTICITY_SCALE}")
        except (ValueError, EOFError):
            print("  Invalid input")

    # Get failure tags
    print(f"\nFailure tags (comma-separated, or press Enter for none):")
    for i, tag in enumerate(FAILURE_TAGS):
        print(f"  {i+1}. {tag}")
    tag_input = input("Tags: ").strip()

    selected_tags = []
    if tag_input:
        try:
            indices = [int(x.strip()) - 1 for x in tag_input.split(",")]
            selected_tags = [FAILURE_TAGS[i] for i in indices if 0 <= i < len(FAILURE_TAGS)]
        except ValueError:
            # Allow direct tag names too
            selected_tags = [t.strip() for t in tag_input.split(",") if t.strip() in FAILURE_TAGS]

    # Get specific failure description
    failure_desc = input("\nDescribe the specific failure (or press Enter to skip): ").strip()

    return {
        "result_id": result["prompt_id"],
        "session_id": result.get("session_id", ""),
        "validator": validator_name,
        "prompt": result["prompt"],
        "date_context": result["date_context"],
        "category": result["category"],
        "response_text": response["response_text"],
        "authenticity_rating": rating,
        "failure_tags": selected_tags,
        "failure_description": failure_desc,
        "annotated_at": datetime.utcnow().isoformat(),
    }


def run_annotation_session(results: list[dict], validator_name: str, output_dir: Path):
    """Run an interactive annotation session."""
    print(f"\n{'='*65}")
    print(f"DARWIN ARCHAEOLOGIST — HUMAN VALIDATION SESSION")
    print(f"Validator: {validator_name}")
    print(f"Results to annotate: {len(results)}")
    print(f"{'='*65}")
    print("\nYou will be shown model-generated responses to prompts about Darwin.")
    print("Rate each for authenticity and identify specific failures.")
    print("Press Ctrl+C at any time to save and exit.\n")

    annotations = []
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, result in enumerate(results, 1):
            print(f"\n[{i}/{len(results)}]", end="")
            annotation = annotate_result(result, validator_name)
            if annotation:
                annotations.append(annotation)
                print(f"  ✓ Annotated (rating: {annotation['authenticity_rating']}/5)")

    except KeyboardInterrupt:
        print(f"\n\n  Session interrupted. Saving {len(annotations)} annotations...")

    if not annotations:
        return

    # Save annotations
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"annotations_{validator_name}_{timestamp}.json"
    out_path.write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved → {out_path}")

    # Print summary
    ratings = [a["authenticity_rating"] for a in annotations]
    avg_rating = sum(ratings) / len(ratings)
    print(f"\n  Annotations: {len(annotations)}")
    print(f"  Average authenticity: {avg_rating:.2f}/5")

    tag_counts = {}
    for a in annotations:
        for tag in a.get("failure_tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    if tag_counts:
        print(f"  Most common failures: {sorted(tag_counts.items(), key=lambda x: -x[1])[:3]}")


def export_to_csv(results_dir: Path, output_path: Path):
    """Export all annotations to CSV for analysis."""
    annotation_files = list(results_dir.glob("annotations_*.json"))
    if not annotation_files:
        print("No annotation files found.")
        return

    all_annotations = []
    for f in annotation_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        all_annotations.extend(data if isinstance(data, list) else [data])

    if not all_annotations:
        return

    fieldnames = [
        "result_id", "session_id", "validator", "category",
        "prompt", "date_context", "authenticity_rating",
        "failure_tags", "failure_description", "annotated_at",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ann in all_annotations:
            ann["failure_tags"] = "|".join(ann.get("failure_tags", []))
            writer.writerow(ann)

    print(f"Exported {len(all_annotations)} annotations → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--session", help="Session ID to annotate")
    parser.add_argument("--validator", default="anonymous", help="Your name/identifier")
    parser.add_argument("--export", action="store_true", help="Export annotations to CSV")
    parser.add_argument("--output", type=Path, help="Output path for CSV export")
    args = parser.parse_args()

    if args.export:
        out = args.output or args.results / "annotations.csv"
        export_to_csv(args.results, out)
        return

    results = load_results(args.results, args.session)
    if not results:
        print("No results to annotate. Run elicitor.py first.")
        sys.exit(1)

    run_annotation_session(results, args.validator, args.results)


if __name__ == "__main__":
    main()
