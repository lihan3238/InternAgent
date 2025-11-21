"""Standalone harness to run PaperGenerationAgent on AutoForecast artifacts."""
import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv  # type: ignore[import-not-found]

from internagent.mas.agents.agent_factory import AgentFactory
from internagent.mas.agents.paper_generation_agent import (
    PaperGenerationAgent,
    _discover_idea_result_dirs,
)
from internagent.mas.models.model_factory import ModelFactory

AUTOFORECAST_IDEA_RUNS = [
    "20251114_223446_Enhanced Adversarial Temporal Convolution-Transfor",
    "20251118_134600_Dynamic Attention Transformer for Energy Forecasti",
]


class _DummyModel:
    async def generate(self, prompt: str):
        return {
            "text": (
                "\\section{Introduction}\nThis is a placeholder draft body produced by the dummy model.\n"
                "\\section{Method}\nDetails are omitted in offline mode; replace with real LLM output.\n"
                "\\section{Conclusion}\nDemo completed."
            )
        }

    async def call(self, prompt: str):
        return {"text": "ack"}


def _build_dummy_agent(agent_conf: Dict[str, Any]) -> Tuple[PaperGenerationAgent, _DummyModel]:
    conf = dict(agent_conf)
    dummy = _DummyModel()
    conf.setdefault("small_model", dummy)
    conf.setdefault("big_model", dummy)
    conf.setdefault("vlm_model", None)
    return PaperGenerationAgent(dummy, config=conf), dummy


def _normalize_dirs(candidates: List[str], root_output: str) -> List[str]:
    """Expand relative directory hints into absolute paths if they exist."""
    normalized: List[str] = []
    root_output = os.path.abspath(root_output)
    for entry in candidates:
        if not entry:
            continue
        probe_order = []
        if os.path.isabs(entry):
            probe_order.append(entry)
        else:
            probe_order.extend(
                [
                    entry,
                    os.path.join(root_output, entry),
                    os.path.abspath(entry),
                ]
            )
        match = None
        for probe in probe_order:
            if probe and os.path.exists(probe):
                match = os.path.abspath(probe)
                break
        if match and match not in normalized:
            normalized.append(match)
    return normalized


def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _load_task_context(task_dir: str, ideas_path: str, idea_dirs: List[str]) -> Dict[str, Any]:
    prompt_path = os.path.join(task_dir, "prompt.json")
    task_meta = _load_json(prompt_path) or {}

    ideas_payload = _load_json(ideas_path)
    if isinstance(ideas_payload, dict):
        ideas = ideas_payload.get("ideas") or ideas_payload.get("hypotheses") or []
    elif isinstance(ideas_payload, list):
        ideas = ideas_payload
    else:
        ideas = []

    results: List[Dict[str, Any]] = []
    for idea_dir in idea_dirs:
        idea_name = os.path.basename(os.path.normpath(idea_dir))
        try:
            for entry in os.listdir(idea_dir):
                if not entry.startswith("run_"):
                    continue
                run_path = os.path.join(idea_dir, entry)
                if not os.path.isdir(run_path):
                    continue
                final_info_path = os.path.join(run_path, "final_info.json")
                record = {
                    "idea_name": idea_name,
                    "artifact_dir": idea_dir,
                    "run_dir": run_path,
                    "final_info": _load_json(final_info_path),
                }
                results.append(record)
        except Exception:
            continue

    return {
        "task": {
            "name": task_meta.get("task_name") or task_meta.get("task") or os.path.basename(task_dir),
            "description": task_meta.get("task_description", ""),
            "domain": task_meta.get("domain", ""),
        },
        "ideas": ideas,
        "results": results,
    }


def _default_abstract(root_output: str) -> str:
    candidates = [
        os.path.join(root_output, "experiment_summary.md"),
        os.path.join(root_output, "experiment_summary_iteration_1.md"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            if text:
                first_block = text.split("\n\n")[0]
                return first_block[:800]
        except Exception:
            continue
    return ""


def run_agent(args: argparse.Namespace) -> Dict[str, Any]:
    load_dotenv()

    with open(args.config, "r", encoding="utf-8") as cfg:
        config = yaml.safe_load(cfg) or {}

    paper_conf = dict(config.get("agents", {}).get("paper_generation", {}) or {})
    paper_conf["_global_config"] = config

    model_factory = ModelFactory()
    agent = None
    creation_error = None

    dummy_model_for_params = None

    if args.use_dummy_models:
        print("[test_paper_generation] Using built-in dummy models (offline mode).")
    else:
        try:
            agent = AgentFactory.create_agent("paper_generation", paper_conf, model_factory)
        except Exception as exc:
            creation_error = exc
            print(f"[test_paper_generation] Failed to create real models ({exc}); falling back to dummy mode.")

    if agent is None:
        agent, dummy_model_for_params = _build_dummy_agent(paper_conf)


    root_output_dir = os.path.abspath(args.root_output)
    os.makedirs(root_output_dir, exist_ok=True)

    idea_hints = args.idea_dirs or AUTOFORECAST_IDEA_RUNS
    idea_dirs = _normalize_dirs(list(idea_hints), root_output_dir)
    if not idea_dirs:
        idea_dirs = _discover_idea_result_dirs(root_output_dir)
    if not idea_dirs:
        raise RuntimeError("No idea result directories found. Provide --idea-dirs or ensure run_* folders exist.")

    print("[test_paper_generation] Using idea result directories:")
    for entry in idea_dirs:
        print(f"  - {entry}")

    ideas_path = os.path.join(root_output_dir, "ideas.json")
    context = _load_task_context(args.task_dir, ideas_path, idea_dirs)

    abstract = args.abstract or _default_abstract(root_output_dir)

    task_name = os.path.basename(args.task_dir.rstrip("/\\")) or "AutoForecast"

    params = {
        "root_output_dir": root_output_dir,
        "iteration": args.iteration,
        "idea_result_dirs": idea_dirs,
        "task_name": task_name,
    }
    if abstract:
        params["abstract"] = abstract
    if args.use_dummy_models or creation_error:
        if dummy_model_for_params is None:
            _, dummy_model_for_params = _build_dummy_agent(paper_conf)
        params.setdefault("big_model", dummy_model_for_params)
        params.setdefault("small_model", dummy_model_for_params)

    print("[test_paper_generation] Starting PaperGenerationAgent execution... this can take a few minutes if remote LLMs are used.")
    return asyncio.run(agent.execute(context, params))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PaperGenerationAgent with AutoForecast artifacts")
    parser.add_argument("--root-output", default=os.path.join("results", "AutoForecast"), help="Directory containing AutoForecast outputs")
    parser.add_argument("--task-dir", default=os.path.join("tasks", "AutoForecast"), help="Task directory hosting prompt.json")
    parser.add_argument("--config", default="config/default_config.yaml", help="Config file providing model settings")
    parser.add_argument("--iteration", default="manual-test", help="Identifier for the LaTeX workspace suffix")
    parser.add_argument("--abstract", default="", help="Optional abstract to seed the paper agent")
    parser.add_argument("--idea-dirs", nargs="*", help="Specific idea directories to include (absolute or relative)")
    parser.add_argument(
        "--use-dummy-models",
        action="store_true",
        help="Run with lightweight local dummy models instead of calling external LLM providers",
    )
    args = parser.parse_args()

    result = run_agent(args)
    latex_dir = result.get("latex_dir")
    if latex_dir:
        print(f"LaTeX project: {latex_dir}")
    if result.get("pdf_path"):
        print(f"PDF path: {result['pdf_path']}")
    if result.get("meta_path"):
        print(f"Meta file: {result['meta_path']}")


if __name__ == "__main__":
    main()
