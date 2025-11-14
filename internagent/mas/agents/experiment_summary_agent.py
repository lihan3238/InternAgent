"""
Experiment Summary Agent

Generates a structured analysis of experimental runs, extracting core ideas,
comparing outcomes across attempts and baseline, linking to literature, and
proposing next-step plans and improvements.
"""

import json
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent, AgentExecutionError
from ..workflow.data_type import WorkflowSession
from ...prompts import EXPERIMENT_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class ExperimentSummaryAgent(BaseAgent):
    def __init__(self, model, config: Dict[str, Any]):
        super().__init__(model, config)
        self.agent_type = "experiment_summary"

    async def execute(self, context: Dict[str, Any], params: Dict[str, Any] = None) -> Dict[str, Any]:
        if context is None:
            raise AgentExecutionError("Context is required for experiment summary")

        # Extract context
        task_info: Dict[str, Any] = context.get("task", {})
        ideas: List[Dict[str, Any]] = context.get("ideas", [])
        results: List[Dict[str, Any]] = context.get("results", [])
        session_traj: Dict[str, Any] = context.get("session_traj", {})

        # Build compact, model-friendly payloads
        def _unwrap_idea(raw_idea: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(raw_idea, dict):
                return {}
            if isinstance(raw_idea.get("refined_method_details"), dict):
                base = dict(raw_idea["refined_method_details"])
                if "references" not in base and raw_idea.get("references"):
                    base["references"] = raw_idea.get("references", [])
                return base
            if isinstance(raw_idea.get("method_details"), dict):
                base = dict(raw_idea["method_details"])
                if "references" not in base and raw_idea.get("references"):
                    base["references"] = raw_idea.get("references", [])
                return base
            return raw_idea

        def _extract_plan_block(source: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(source, dict):
                return {}
            plan_candidates = [
                source.get("experiment_plan"),
                source.get("execution_plan"),
                source.get("implementation_plan"),
                source.get("plan"),
                source.get("experimental_plan"),
                source.get("plan_summary"),
            ]
            for plan in plan_candidates:
                if plan:
                    if isinstance(plan, str):
                        return {"overview": plan}
                    if isinstance(plan, dict):
                        return plan
            if source.get("implementation_steps"):
                return {"implementation_steps": source.get("implementation_steps")}
            return {}

        compact_ideas = []
        idea_lookup = {}
        for idea in ideas or []:
            normalized = _unwrap_idea(idea)
            if not normalized:
                continue
            entry = {
                "name": normalized.get("name") or normalized.get("title") or "",
                "title": normalized.get("title") or normalized.get("name") or "",
                "description": normalized.get("description") or normalized.get("content") or "",
                "statement": normalized.get("statement", ""),
                "method": normalized.get("method", ""),
                "baseline_summary": normalized.get("baseline_summary", ""),
                "references": (normalized.get("references") or [])[:6],
                "experiment_plan": _extract_plan_block(normalized),
                "expected_metrics": normalized.get("expected_metrics") or normalized.get("metrics"),
                "validation": normalized.get("validation") or normalized.get("evaluation"),
            }
            compact_ideas.append(entry)
            key_name = (entry["name"] or entry["title"]).strip().lower()[:50]
            if key_name:
                idea_lookup[key_name] = normalized

        experiment_runs = []
        for r in results or []:
            idea_name = r.get("idea_name") or ""
            normalized_key = idea_name.strip().lower()[:50]
            plan_source = idea_lookup.get(normalized_key, {})
            run_history = r.get("run_history") if isinstance(r.get("run_history"), dict) else {}
            timeline = []
            baseline_metrics = run_history.get("baseline")
            if baseline_metrics:
                timeline.append({
                    "run_id": "baseline",
                    "status": "baseline",
                    "metrics": baseline_metrics
                })
            for run_entry in run_history.get("runs", []):
                timeline.append(run_entry)
            experiment_runs.append({
                "idea_name": idea_name,
                "status": "success" if r.get("success") else "failed",
                "error": r.get("error", ""),
                "artifact_dir": r.get("artifact_dir"),
                "idea_details": r.get("idea_details") or {},
                "plan_context": {
                    "source_plan": _extract_plan_block(plan_source),
                    "method_statement": plan_source.get("statement"),
                    "expected_metrics": plan_source.get("expected_metrics") or plan_source.get("metrics"),
                },
                "run_history": {
                    "baseline": run_history.get("baseline", {}),
                    "runs": run_history.get("runs", []),
                    "timeline": timeline,
                }
            })

        # Minimal trajectory history if available
        history = {}
        try:
            if isinstance(session_traj, dict):
                history = {
                    "iterations_completed": session_traj.get("iterations_completed"),
                    "top_hypotheses": session_traj.get("top_hypotheses"),
                    "feedback_count": len(session_traj.get("feedback_history", [])),
                }
        except Exception:
            history = {}

        # JSON that the model will see
        packed = {
            "task": task_info,
            "ideas": compact_ideas,
            "experiment_runs": experiment_runs,
            "history": history,
        }

        output_schema = {
            "type": "object",
            "properties": {
                "idea_analyses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "idea_name": {"type": "string"},
                            "plan_brief": {"type": "string"},
                            "run_analyses": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "run_id": {"type": "string"},
                                        "status": {"type": "string"},
                                        "variables": {"type": "string"},
                                        "metrics_delta": {"type": "string"},
                                        "diagnosis": {"type": "string"},
                                        "actions": {"type": "string"},
                                    },
                                    "required": ["run_id", "status", "diagnosis"]
                                }
                            },
                            "idea_takeaways": {"type": "string"},
                            "idea_next_steps": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["idea_name", "run_analyses", "idea_takeaways"]
                    }
                },
                "core_idea": {"type": "string"},
                "failure_overview": {"type": "string"},
                "result_analysis": {"type": "string"},
                "failures_and_causes": {"type": "array", "items": {"type": "string"}},
                "literature_links": {"type": "array", "items": {"type": "string"}},
                "next_plan": {
                    "type": "object",
                    "properties": {
                        "objective": {"type": "string"},
                        "milestones": {"type": "array", "items": {"type": "string"}},
                        "experiments": {"type": "array", "items": {"type": "string"}},
                        "metrics": {"type": "array", "items": {"type": "string"}},
                        "risks": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["objective", "milestones"]
                },
                "improvement_directions": {"type": "array", "items": {"type": "string"}},
                "summary_markdown": {"type": "string"}
            },
            "required": [
                "idea_analyses", "core_idea", "failure_overview", "result_analysis",
                "next_plan", "improvement_directions", "summary_markdown"
            ]
        }

        prompt = EXPERIMENT_SUMMARY_PROMPT.format(
            CONTEXT_JSON=json.dumps(packed, ensure_ascii=False, indent=2)
        )

        default = {
            "idea_analyses": [],
            "core_idea": "",
            "failure_overview": "",
            "result_analysis": "",
            "failures_and_causes": [],
            "literature_links": [],
            "next_plan": {"objective": "", "milestones": [], "experiments": [], "metrics": [], "risks": []},
            "improvement_directions": [],
            "summary_markdown": ""
        }

        try:
            result = await self.model.generate_json(
                prompt=prompt,
                schema=output_schema,
                system_prompt=self.system_prompt or "You are a rigorous research analyst.",
                temperature=self.config.get("temperature", 0.3),
                default=default,
            )
            return result
        except Exception as e:
            logger.error(f"Experiment summary generation failed: {e}")
            raise AgentExecutionError(str(e))
