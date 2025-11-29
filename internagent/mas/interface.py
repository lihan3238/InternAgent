"""
InternAgent Interface

This module implements the main interface for the InternAgent system.
The InternAgentInterface manages system lifecycle, session coordination,
and provides a clean API for research workflow interactions.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List

from .models.model_factory import ModelFactory
from .agents.agent_factory import AgentFactory
from .memory.memory_manager import MemoryManager
from .workflow.orchestration_agent import OrchestrationAgent
from .workflow.data_type import WorkflowState

logger = logging.getLogger(__name__)


class InternAgentInterface:
    """
    Main interface for the InternAgent system.

    This class provides a high-level API for managing the research workflow system,
    including session management, orchestration agent coordination, and system lifecycle.
    It serves as the primary entry point for external applications.
    """

    def __init__(self, config_path: str = None, config: Dict[str, Any] = None, work_dir: str = 'example', task_name: str = None):
        """
        Initialize the InternAgent interface.

        Args:
            config_path: Path to the configuration file
            config: Configuration dictionary (takes precedence over config_path)
            work_dir: Working directory for the system
            task_name: Name of the task (from --task parameter)
        """
        self.work_dir = work_dir
        self.task_name = task_name or "DefaultTask"
        self.config = self._load_config(config_path, config)
        self.config['config_path'] = config_path
        self.config['work_dir'] = self.work_dir
        self.config['task_name'] = self.task_name

        # Core components
        self.model_factory = ModelFactory()
        self.memory_manager = self._init_memory_manager()
        self.agent_factory = AgentFactory()

        # Create specialized agents
        self.agents = self.agent_factory.create_all_agents(
            config=self.config,
            model_factory=self.model_factory
        )

        # Initialize orchestration agent
        self.orchestration_agent = self._init_orchestration_agent()

        # Session tracking
        self.active_sessions = {}

        # System state
        self.system_ready = False

        logger.info("InternAgent interface initialized")

    async def startup(self) -> None:
        """Initialize system components and mark as ready."""
        try:
            await self.memory_manager.startup()
            self.system_ready = True
            logger.info("InternAgent system started successfully")
        except Exception as e:
            self.system_ready = False
            logger.error(f"Error during system startup: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Clean shutdown of system components."""
        try:
            if hasattr(self.memory_manager, 'shutdown'):
                await self.memory_manager.shutdown()
            self.system_ready = False
            logger.info("InternAgent system shut down successfully")
        except Exception as e:
            logger.error(f"Error during system shutdown: {str(e)}")
            raise

    async def create_session(self,
                           goal_description: str,
                           domain: str,
                           background: str = "",
                           ref_code_path: str = "",
                           constraints: List[str] = None) -> str:
        """
        Create a new research session.

        Args:
            goal_description: Description of the research goal
            domain: Scientific domain for the research
            background: Background information for context
            ref_code_path: Path to reference code (optional)
            constraints: List of constraints for the session

        Returns:
            Session ID for the created session

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()

        session_id = await self.orchestration_agent.create_session(
            goal_description=goal_description,
            domain=domain,
            background=background,
            ref_code_path=ref_code_path,
            constraints=constraints or []
        )

        # Track session
        self._track_session(session_id, goal_description, domain, ref_code_path)

        logger.info(f"Created session {session_id}: {goal_description}")
        return session_id

    async def run_session(self,
                        session_id: str,
                        status_callback: Optional[Callable[[str, str, str], None]] = None) -> Dict[str, Any]:
        """
        Execute a research session.

        Args:
            session_id: ID of the session to run
            status_callback: Optional callback for state change notifications
                           Signature: (session_id, old_state, new_state) -> None

        Returns:
            Current session status

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()

        # Create state change handler
        async def on_state_change(session, old_state, new_state):
            self._update_session_tracking(session_id, new_state.value)

            if status_callback:
                try:
                    await status_callback(session_id, old_state.value, new_state.value)
                except Exception as e:
                    logger.error(f"Error in status callback: {str(e)}")

        # Execute session
        await self.orchestration_agent.run_session(
            session_id=session_id,
            on_state_change=on_state_change
        )

        return await self.get_session_status(session_id)

    async def add_feedback(self,
                         session_id: str,
                         feedback: dict,
                         target_idea_ids: List[str] = None,
                         auto_resume: bool = True) -> Dict[str, Any]:
        """
        Add feedback to a session and optionally resume execution.

        Args:
            session_id: Session ID
            feedback: Feedback content
            target_idea_ids: Specific ideas the feedback applies to
            auto_resume: Whether to automatically resume session after feedback

        Returns:
            Updated session status

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()

        await self.orchestration_agent.add_feedback(
            session_id=session_id,
            feedback=feedback,
            target_idea_ids=target_idea_ids
        )

        self._update_session_tracking(session_id)

        # Auto-resume if requested and session is waiting for feedback
        if auto_resume:
            status = await self.get_session_status(session_id)
            if status.get("state") == WorkflowState.REFLECTING.value:
                await self.resume_session(session_id)

        return await self.get_session_status(session_id)

    async def resume_session(self, session_id: str) -> Dict[str, Any]:
        """
        Resume a paused session.

        Args:
            session_id: Session ID to resume

        Returns:
            Updated session status

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()

        await self.orchestration_agent.run_session(session_id)
        self._update_session_tracking(session_id)

        return await self.get_session_status(session_id)

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get current status of a session.

        Args:
            session_id: Session ID

        Returns:
            Status dictionary with session information

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()
        return await self.orchestration_agent.get_session_status(session_id)

    async def get_top_ideas(self,
                               session_id: str,
                               include_all: bool = False) -> List[Dict[str, Any]]:
        """
        Get top ideas from a session.

        Args:
            session_id: Session ID
            include_all: Whether to include ideas from all iterations

        Returns:
            List of hypothesis dictionaries

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()
        return await self.orchestration_agent.get_top_ideas(
            session_id=session_id,
            include_all=include_all
        )

    async def get_all_ideas(self,
                               session_id: str,
                               limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get all ideas from a session.

        Args:
            session_id: Session ID
            limit: Maximum number of ideas to return

        Returns:
            List of hypothesis dictionaries

        Raises:
            RuntimeError: If system is not ready
        """
        self._ensure_system_ready()
        return await self.orchestration_agent.get_ideas(
            session_id=session_id,
            limit=limit,
            include_all=True
        )

    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of all active sessions.

        Returns:
            List of session information dictionaries
        """
        return list(self.active_sessions.values())

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and status.

        Returns:
            System information dictionary
        """
        return {
            "ready": self.system_ready,
            "work_dir": self.work_dir,
            "active_sessions": len(self.active_sessions),
            "agents_loaded": len(self.agents),
            "config_version": self.config.get("version", "unknown")
        }

    # Private helper methods
    def _ensure_system_ready(self) -> None:
        """Ensure system is ready for operations."""
        if not self.system_ready:
            raise RuntimeError("System is not ready. Call startup() first.")

    def _track_session(self, session_id: str, goal: str, domain: str, ref_code_path: str) -> None:
        """Add session to tracking."""
        self.active_sessions[session_id] = {
            "id": session_id,
            "goal": goal,
            "domain": domain,
            "ref_code_path": ref_code_path,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "state": WorkflowState.INITIAL.value
        }

    def _update_session_tracking(self, session_id: str, state: str = None) -> None:
        """Update session tracking information."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_updated"] = datetime.now().isoformat()
            if state:
                self.active_sessions[session_id]["state"] = state

    def _init_memory_manager(self) -> MemoryManager:
        """Initialize memory manager from configuration."""
        from .memory.memory_manager import create_memory_manager
        return create_memory_manager(self.config)

    def _init_orchestration_agent(self) -> OrchestrationAgent:
        """Initialize orchestration agent with all components."""
        return OrchestrationAgent(
            config=self.config,
            memory_manager=self.memory_manager,
            model_factory=self.model_factory,
            agent_registry=self.agents
        )

    def _load_config(self, config_path: Optional[str], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Load configuration from file or dictionary.

        Args:
            config_path: Path to configuration file
            config: Configuration dictionary

        Returns:
            Loaded configuration dictionary
        """
        # Use provided config if available
        if config is not None:
            return config

        # Load from file if path provided and file exists
        if config_path and os.path.exists(config_path):
            logger.info(f"Attempting to load config from: {config_path}")
            try:
                if config_path.endswith(('.yaml', '.yml')):
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    logger.info(f"Successfully loaded YAML config with keys: {list(config_data.keys()) if config_data else 'None'}")
                    return config_data
                else:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    logger.info(f"Successfully loaded JSON config with keys: {list(config_data.keys()) if config_data else 'None'}")
                    return config_data
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {str(e)}")
        elif config_path:
            logger.warning(f"Config path provided but file doesn't exist: {config_path}")
        else:
            logger.info("No config path provided")

        # Load default configuration
        default_config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "default.yaml")
        default_config_path = os.path.abspath(default_config_path)
        logger.info(f"Loading default configuration from: {default_config_path}")

        try:
            import yaml
            with open(default_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            logger.info(f"Successfully loaded default config with keys: {list(config_data.keys()) if config_data else 'None'}")
            return config_data
        except Exception as e:
            logger.error(f"Failed to load default config from {default_config_path}: {str(e)}")
            raise RuntimeError(f"Could not load default configuration: {str(e)}")

