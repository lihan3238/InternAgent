import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from internagent.mas.models.model_factory import ModelFactory


def _read_json_safe(path: str) -> Any:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_task_meta(task_dir: str) -> Dict[str, Any]:
    prompt_path = os.path.join(task_dir, "prompt.json")
    payload = _read_json_safe(prompt_path) or {}
    return {
        "name": os.path.basename(task_dir.rstrip("/\\")) or "Task",
        "description": payload.get("task_description", ""),
        "domain": payload.get("domain", ""),
        "constraints": payload.get("constraints", []),
        "background": payload.get("background", ""),
    }


def _normalize_ideas(raw_ideas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ideas = []
    for idea in raw_ideas or []:
        details = idea.get("refined_method_details") or idea.get("method_details") or idea
        ideas.append(
            {
                "id": details.get("id") or idea.get("id"),
                "name": details.get("name") or details.get("title") or "idea",
                "title": details.get("title") or details.get("name") or "idea",
                "description": details.get("description") or idea.get("description") or idea.get("text") or "",
                "method": details.get("method", ""),
                "raw": idea,
            }
        )
    return ideas


def _load_ideas_from_paths(candidate_paths: List[str]) -> List[Dict[str, Any]]:
    for path in candidate_paths:
        payload = _read_json_safe(path)
        if not payload:
            continue
        if isinstance(payload, list):
            return _normalize_ideas(payload)
        if isinstance(payload, dict):
            if "ideas" in payload:
                return _normalize_ideas(payload.get("ideas") or [])
            if "hypotheses" in payload:
                return _normalize_ideas(payload.get("hypotheses") or [])
    return []


def _collect_results(iteration_results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    results = []
    idea_result_dirs: List[str] = []
    for entry in iteration_results or []:
        artifact_dir = entry.get("artifact_dir")
        if artifact_dir and os.path.isdir(artifact_dir):
            idea_result_dirs.append(artifact_dir)
        run_history = entry.get("run_history") or {}
        results.append(
            {
                "idea_label": entry.get("idea_name") or entry.get("idea_label"),
                "idea_name": entry.get("idea_name"),
                "success": entry.get("success", False),
                "artifact_dir": artifact_dir,
                "run_history": run_history,
                "error": entry.get("error"),
                "notes": entry.get("error"),
            }
        )
    return results, list(dict.fromkeys(idea_result_dirs))


def _model_spec_from_global_conf(global_conf: Dict[str, Any], temperature: float = None, max_tokens: int = None) -> Dict[str, Any]:
    models_conf = global_conf.get("models", {}) if isinstance(global_conf, dict) else {}
    provider = models_conf.get("default_provider", "openai")
    provider_conf = models_conf.get(provider, {}) if isinstance(models_conf.get(provider, {}), dict) else {}
    spec = {
        "provider": provider,
        "default_provider": provider,
        "models": models_conf,
    }
    if provider_conf:
        spec.update({k: v for k, v in provider_conf.items() if k not in spec})
    if temperature is not None:
        spec["temperature"] = temperature
    if max_tokens is not None:
        spec["max_tokens"] = max_tokens
    return spec


def build_paper_inputs(
    task_dir: str,
    iteration_label: str,
    mas_iter_dir: str,
    exp_iter_dir: str,
    iteration_results: List[Dict[str, Any]],
    top_ideas: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    task_meta = _load_task_meta(task_dir)
    ideas = _normalize_ideas(top_ideas)
    if not ideas:
        candidate_ideas = [
            os.path.join(mas_iter_dir, f"ideas_iter{iteration_label}.json"),
            os.path.join(mas_iter_dir, "ideas.json"),
            os.path.join(os.path.dirname(mas_iter_dir), "latest", "ideas.json"),
        ]
        ideas = _load_ideas_from_paths(candidate_ideas)

    results, idea_result_dirs = _collect_results(iteration_results)

    context = {
        "task": task_meta,
        "ideas": ideas,
        "results": results,
    }

    params = {
        "root_output_dir": os.path.join(exp_iter_dir, "paper_outputs"),
        "iteration": iteration_label,
        "idea_result_dirs": idea_result_dirs,
    }
    return context, params


def _preload_paper_agent():
    """Ensure paper_agent importable from vendored path or env hint."""
    try:
        import internagent.mas.agents.paper_agent  # noqa: F401
        return
    except Exception:
        pass

    candidates = []
    env_path = os.environ.get("PAPER_AGENT_PATH")
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "internagent" / "mas" / "agents" / "paper_agent")
    candidates.append(repo_root / "paper_agent")
    candidates.append(repo_root.parent / "paper_agent")

    for cand in candidates:
        if cand and cand.exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            try:
                import internagent.mas.agents.paper_agent  # noqa: F401
                return
            except Exception:
                continue
    raise ImportError("paper_agent not found; ensure it exists under internagent/mas/agents/paper_agent or set PAPER_AGENT_PATH.")


async def generate_paper_async(context: Dict[str, Any], params: Dict[str, Any], global_conf: Dict[str, Any] = None) -> Dict[str, Any]:
    try:
        _preload_paper_agent()
        from internagent.mas.agents.paper_agent.paper_generation_agent import PaperGenerationAgent
    except Exception as exc:
        raise RuntimeError(f"paper_agent not available: {exc}") from exc

    agent_conf = dict(params or {})
    agent_conf["output_dir"] = params.get("root_output_dir")

    global_conf = global_conf or {}
    big_spec = _model_spec_from_global_conf(global_conf)
    small_spec = _model_spec_from_global_conf(global_conf, temperature=0.5)
    agent_conf.setdefault("big_model", big_spec)
    agent_conf.setdefault("small_model", small_spec)

    main_model = None
    try:
        merged_conf = dict(big_spec)
        merged_conf.update({"models": global_conf.get("models", {})})
        main_model = ModelFactory.create_model(merged_conf)
    except Exception:
        main_model = None

    agent = PaperGenerationAgent(model=main_model, config=agent_conf)
    return await agent.execute(context, params)


def generate_paper(context: Dict[str, Any], params: Dict[str, Any], global_conf: Dict[str, Any] = None) -> Dict[str, Any]:
    return asyncio.run(generate_paper_async(context, params, global_conf or {}))
