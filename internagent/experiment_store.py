import json
import os
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional


class ExperimentStore(ABC):
    """Abstract experiment store interface used for idea/method/summary persistence."""

    @abstractmethod
    def add_experiment(self, record: Dict[str, Any]) -> str:
        """Persist a new experiment and return its experiment_id."""
        raise NotImplementedError

    @abstractmethod
    def update_status(
        self,
        experiment_id: str,
        status: str,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        run_history: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update execution results, status, and optional summary for an experiment."""
        raise NotImplementedError

    @abstractmethod
    def get(self, experiment_id: str) -> Dict[str, Any]:
        """Retrieve one experiment record."""
        raise NotImplementedError

    @abstractmethod
    def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent experiments for UI / sampling."""
        raise NotImplementedError

    @abstractmethod
    def search_similar_motivation(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Lightweight similarity search over stored motivations/insights."""
        raise NotImplementedError


class JsonExperimentStore(ExperimentStore):
    """Filesystem JSON implementation; one meta.json per experiment."""

    def __init__(self, root_dir: str = "results", experiments_subdir: str = "experiments"):
        self.root_dir = root_dir
        self.experiments_dir = os.path.join(root_dir, experiments_subdir)
        os.makedirs(self.experiments_dir, exist_ok=True)

    def _path(self, experiment_id: str) -> str:
        return os.path.join(self.experiments_dir, f"{experiment_id}.json")

    def add_experiment(self, record: Dict[str, Any]) -> str:
        experiment_id = record.get("experiment_id") or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        record["experiment_id"] = experiment_id
        record.setdefault("status", "Running")
        record.setdefault("created_at", datetime.now().isoformat())
        record.setdefault("updated_at", record["created_at"])

        with open(self._path(experiment_id), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return experiment_id

    def _load(self, experiment_id: str) -> Dict[str, Any]:
        try:
            with open(self._path(experiment_id), "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def update_status(
        self,
        experiment_id: str,
        status: str,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None,
        run_history: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = self._load(experiment_id)
        if not meta:
            return
        meta["status"] = status
        if error:
            meta["error"] = error
        if artifacts is not None:
            meta["artifacts"] = artifacts
        if run_history is not None:
            meta["run_history"] = run_history
        if summary is not None:
            meta["summary"] = summary
        meta["updated_at"] = datetime.now().isoformat()
        with open(self._path(experiment_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def get(self, experiment_id: str) -> Dict[str, Any]:
        return self._load(experiment_id)

    def list_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        entries = []
        if not os.path.exists(self.experiments_dir):
            return entries
        for item in sorted(os.listdir(self.experiments_dir), reverse=True):
            meta_path = self._path(item)
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    entries.append(json.load(f))
            except Exception:
                continue
            if len(entries) >= limit:
                break
        return entries

    def search_similar_motivation(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Naive similarity using difflib on combined motivation/design_insight."""
        if not text:
            return []
        candidates = []
        for meta in self.list_recent(limit=200):
            summary_val = meta.get("summary")
            if isinstance(summary_val, str):
                summary_text = summary_val
            elif isinstance(summary_val, dict):
                summary_text = json.dumps(summary_val, ensure_ascii=False)
            else:
                summary_text = ""
            corpus = (
                (meta.get("motivation") or "")
                + "\n"
                + (meta.get("design_insight") or "")
                + "\n"
                + summary_text
            )
            score = SequenceMatcher(None, text, corpus).ratio()
            candidates.append(
                {
                    "experiment_id": meta.get("experiment_id"),
                    "idea_name": meta.get("idea_name"),
                    "status": meta.get("status"),
                    "score": score,
                    "motivation": meta.get("motivation"),
                    "design_insight": meta.get("design_insight"),
                    "summary": meta.get("summary"),
                }
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]


def create_experiment_store(config: Dict[str, Any]) -> ExperimentStore:
    """Factory using config/CLI overrides."""
    store_cfg = (config or {}).get("store", {}) if isinstance(config, dict) else {}
    backend = store_cfg.get("backend", "json")
    root_dir = store_cfg.get("root_dir", "results")
    experiments_subdir = store_cfg.get("experiments_subdir", "experiments")

    # Currently only json backend is implemented; hook for future mongo.
    if backend != "json":
        backend = "json"

    return JsonExperimentStore(root_dir=root_dir, experiments_subdir=experiments_subdir)
