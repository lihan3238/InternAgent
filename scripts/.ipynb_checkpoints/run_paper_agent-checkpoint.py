import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `internagent` and `paper_agent` are importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from internagent.paper_bridge import build_paper_inputs, generate_paper


def _read_json(path: str):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def main():
    parser = argparse.ArgumentParser(description="Run paper_agent on a finished InternAgent iteration.")
    parser.add_argument("--task_dir", required=True, help="Task directory (e.g., tasks/AutoForecast)")
    parser.add_argument("--iteration", required=True, type=int, help="Iteration number to use")
    parser.add_argument(
        "--output_root",
        default=None,
        help="Root of results; defaults to results/<task_name>",
    )
    args = parser.parse_args()

    task_name = os.path.basename(args.task_dir.rstrip("/\\"))
    results_root = args.output_root or os.path.join("results", task_name)
    iter_label = f"iteration_{args.iteration}"
    mas_iter_dir = os.path.join(results_root, "mas", "iterations", iter_label)
    exp_iter_dir = os.path.join(results_root, "experiments", "iterations", iter_label)

    summary_path = os.path.join(mas_iter_dir, f"discovery_summary_iter{args.iteration}.json")
    if not os.path.exists(summary_path):
        latest_summary = os.path.join(results_root, "mas", "latest", "discovery_summary.json")
        summary_path = latest_summary if os.path.exists(latest_summary) else summary_path

    summary_payload = _read_json(summary_path)
    iteration_results = summary_payload.get("results", [])

    ideas_payload = summary_payload.get("ideas") or []
    paper_ctx, paper_params = build_paper_inputs(
        task_dir=args.task_dir,
        iteration_label=str(args.iteration),
        mas_iter_dir=mas_iter_dir,
        exp_iter_dir=exp_iter_dir,
        iteration_results=iteration_results,
        top_ideas=ideas_payload,
    )

    print(f"[paper_agent] Building manuscript for {iter_label}")
    result = generate_paper(paper_ctx, paper_params, summary_payload if isinstance(summary_payload, dict) else {})
    print("[paper_agent] Done")
    print(f"latex_dir: {result.get('latex_dir')}")
    print(f"pdf_path : {result.get('pdf_path')}")
    print(f"meta_path: {result.get('meta_path')}")


if __name__ == "__main__":
    main()
