import os.path as osp
import os
import sys
import json
import shutil
from datetime import datetime
import asyncio
import re

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from internagent.mas.interface import InternAgentInterface
from internagent.experiments_utils_aider import perform_experiments as perform_experiments_aider

from internagent.vis import visualize_hypotheses


# ============================================================================
# Idea Generation Config
# ============================================================================
class IdeaGenerator:
    """Handles idea generation using MAS"""
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.interface = InternAgentInterface(args.config, work_dir=args.task_dir, task_name=args.task_name)
        self.session_id = None
        self.status = None
    
    async def load_task(self):
        """Load task and create MAS session"""
        self.logger.info(f"Creating research session for: {self.args.task_dir}")
        
        await self.interface.startup()
        
        task_desc_path = osp.join(self.args.task_dir, "prompt.json")
        if not osp.exists(task_desc_path):
            raise FileNotFoundError(f"Task description not found: {task_desc_path}")
        
        with open(task_desc_path, 'r') as f:
            params = json.load(f)
        
        goal = params.get('task_description')
        domain = params.get('domain')
        background = params.get('background', "")
        constraints = params.get('constraints', [])
        
        if not goal or not domain:
            raise ValueError("Task description and domain are required")
        
        self.session_id = await self.interface.create_session(
            goal_description=goal,
            domain=domain,
            background=background,
            ref_code_path=self.args.ref_code_path,
            constraints=constraints
        )
        
        self.logger.info(f"Session created: {self.session_id}")
    
    async def generate_ideas(self):
        """Run MAS to generate ideas"""
        if self.session_id is None:
            await self.load_task()
        
        await self.interface.startup()
        
        async def status_callback(session_id, old_state, new_state):
            # Agent transition logging is handled by OrchestrationAgent
            pass
        
        while self.status != "completed":
            try:
                full_status = await self.interface.get_session_status(self.session_id)
                self.status = full_status['state']
                iterations = full_status['iterations_completed']
                
                if self.status == "awaiting_feedback":#idea需要人机交互
                    self.logger.info("Session awaiting human feedback")
                    # Idea-level human-in-the-loop: present top ideas with richer context,
                    # write pending files for offline/manual review, then prompt the user.
                    try:
                        loop = asyncio.get_running_loop()
                        top_ideas = await self.interface.get_top_ideas(self.session_id)

                        # Write pending feedback files (one per idea) so non-interactive runs leave artifacts.
                        try:
                            task_dir = osp.join("results", self.args.task_name)
                            os.makedirs(task_dir, exist_ok=True)
                            pending_files = []
                            for idea in top_ideas:
                                pid = idea.get('id')
                                pending_path = osp.join(task_dir, f"pending_feedback_{self.session_id}_{pid}.json")
                                pending_payload = {
                                    "session_id": self.session_id,
                                    "idea_id": pid,
                                                       "title": idea.get('title') or idea.get('name') or idea.get('text') or '',
                                    "description": idea.get('description', ''),
                                    "score": idea.get('score', None) or idea.get('overall_score', None),
                                    "baseline_summary": idea.get('baseline_summary', '') or idea.get('baseline', ''),
                                    "top_references": [ (r.get('title') if isinstance(r, dict) else str(r)) for r in (idea.get('references') or []) ][:5],
                                    "created_at": datetime.now().isoformat()
                                }
                                try:
                                    with open(pending_path, 'w', encoding='utf-8') as pf:
                                        json.dump(pending_payload, pf, indent=2, ensure_ascii=False)
                                    pending_files.append(pending_path)
                                except Exception as e:
                                    self.logger.error(f"Failed to write pending feedback for idea {pid}: {e}")

                            if pending_files:
                                self.logger.info(f"Wrote pending feedback files for ideas: {pending_files}")
                        except Exception as e:
                            self.logger.error(f"Error creating pending feedback files: {e}")

                        def prompt_for_feedback():
                            print("Session requires human feedback. Review the ideas below and enter modification comments.")
                            print("Format: <index>: <comment>  (one per line). Leave blank line to finish.\n")

                            for i, idea in enumerate(top_ideas):
                                # title: prefer explicit title/name, fallback to text
                                title = idea.get('title') or idea.get('name') or idea.get('text') or ''
                                # description may be under different keys
                                desc = idea.get('description') or idea.get('rationale') or ''
                                score = idea.get('score', None) or idea.get('overall_score', None)
                                baseline = idea.get('baseline_summary') or idea.get('baseline') or ''
                                refs = []

                                # Prefer explicit references list
                                if isinstance(idea.get('references'), list) and idea.get('references'):
                                    for r in idea.get('references')[:3]:
                                        if isinstance(r, dict):
                                            refs.append(r.get('title') or r.get('name') or r.get('paper_title') or '')
                                        else:
                                            refs.append(str(r)[:140])
                                else:
                                    # Fallback: try to extract titles from evidence
                                    ev = idea.get('evidence') or idea.get('refine_evidence') or []
                                    for e in ev[:3]:
                                        if isinstance(e, dict):
                                            refs.append(e.get('title') or e.get('paper') or (e.get('content') or '')[:140])
                                        else:
                                            refs.append(str(e)[:140])

                                print(f"[{i}] id:{idea.get('id', '')}  title: {title[:120]}  score: {score}")
                                if baseline:
                                    print(f"    baseline: {baseline[:300]}")
                                if desc:
                                    print(f"    desc: {desc[:400]}")
                                if refs:
                                    print(f"    top refs: {', '.join([r for r in refs if r])}")
                                print("")

                            print("\nEnter comments (or press Enter twice to skip):")

                            lines = []
                            while True:
                                try:
                                    ln = input()
                                except EOFError:
                                    break
                                if not ln.strip():
                                    break
                                lines.append(ln.rstrip())
                            return lines

                        user_lines = await loop.run_in_executor(None, prompt_for_feedback)

                        if user_lines:
                            # Parse lines of the form 'index: comment'
                            comments = []
                            for line in user_lines:
                                if ':' in line:
                                    idx_str, comment = line.split(':', 1)
                                    try:
                                        idx = int(idx_str.strip())
                                    except ValueError:
                                        continue
                                    if 0 <= idx < len(top_ideas):
                                        comments.append({
                                            'id': top_ideas[idx].get('id'),
                                            'comment': comment.strip()
                                        })

                            if comments:
                                feedback = {
                                    'type': 'local',
                                    'content': comments
                                }
                                await self.interface.add_feedback(self.session_id, feedback)
                                self.logger.info(f"Human feedback added for session {self.session_id}: {feedback}")
                            else:
                                # If lines were provided but not parseable as local comments,
                                # send them as a global feedback blob
                                joined = "\n".join(user_lines)
                                feedback = {'type': 'global', 'content': joined}
                                await self.interface.add_feedback(self.session_id, feedback)
                                self.logger.info(f"Human global feedback added for session {self.session_id}")

                        else:
                            # No interactive input provided - fall back to configured offline feedback
                            if self.args.offline_feedback and osp.exists(self.args.offline_feedback):
                                with open(self.args.offline_feedback, "r") as f:
                                    feedback = json.load(f)
                                await self.interface.add_feedback(self.session_id, feedback)
                                self.logger.info(f"Offline feedback applied: {self.args.offline_feedback}")
                            else:
                                self.logger.info("No human feedback provided and no offline feedback file available; continuing without feedback")

                    except Exception as e:
                        # If anything goes wrong with interactive prompting, fall back to offline feedback if available
                        self.logger.error(f"Error collecting human feedback interactively: {e}")
                        if self.args.offline_feedback and osp.exists(self.args.offline_feedback):
                            try:
                                with open(self.args.offline_feedback, "r") as f:
                                    feedback = json.load(f)
                                await self.interface.add_feedback(self.session_id, feedback)
                                self.logger.info(f"Offline feedback applied after interactive failure: {self.args.offline_feedback}")
                            except Exception as e2:
                                self.logger.error(f"Failed to apply offline feedback: {e2}")

                    except Exception as e:
                        # If anything goes wrong with interactive prompting, fall back to offline feedback if available
                        self.logger.error(f"Error collecting human feedback interactively: {e}")
                        if self.args.offline_feedback and osp.exists(self.args.offline_feedback):
                            try:
                                with open(self.args.offline_feedback, "r") as f:
                                    feedback = json.load(f)
                                await self.interface.add_feedback(self.session_id, feedback)
                                self.logger.info(f"Offline feedback applied after interactive failure: {self.args.offline_feedback}")
                            except Exception as e2:
                                self.logger.error(f"Failed to apply offline feedback: {e2}")
                #idea的人机交互结束
                elif self.status == "completed":
                    self.logger.info("Idea generation completed")
                    break
                
                elif self.status == "error":
                    raise RuntimeError("Error in MAS session")
                
                self.logger.info(f"Running session {self.session_id}, iteration {iterations}")
                self.status = await self.interface.run_session(
                    self.session_id,
                    status_callback=status_callback
                )
                
            except Exception as e:
                self.logger.error(f"Error in session: {str(e)}")
                raise
        
        top_ideas = await self.interface.get_top_ideas(self.session_id)
        self.logger.info(f"Generated {len(top_ideas)} top ideas")
        
        # Save and visualize ideas
        # Session files are saved in results/{task_name}/ by the memory manager
        session_json = osp.join("results", self.args.task_name, f"traj_{self.session_id}.json")
        vis_output = osp.join(
            "results", self.args.task_name,
            f"{self.args.task_name}_ideas_{self.session_id}.pdf"
        )
        visualize_hypotheses(session_json, vis_output)
        self.logger.info(f"Visualization saved: {vis_output}")
        
        return top_ideas, session_json


# ============================================================================
# Experiment Execution Module
# ============================================================================
class ExperimentRunner:
    """Handles experiment execution with different backends"""
    
    def __init__(self, args, logger, config=None):
        self.args = args
        self.logger = logger
        self.backend = args.exp_backend
        self.config = config or {}
    
    def _extract_idea_info(self, idea):
        """Extract idea information from different formats"""
        # Try refined_method_details first (from full MAS pipeline)
        if 'refined_method_details' in idea and idea['refined_method_details']:
            details = idea['refined_method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', '')
            }

        # Fall back to method_details (from method development only)
        elif 'method_details' in idea and idea['method_details']:
            details = idea['method_details']
            return {
                'name': details.get('name', 'unnamed_idea'),
                'title': details.get('title', 'Untitled'),
                'description': details.get('description', ''),
                'method': details.get('method', '')
            }

        # Fall back to basic idea structure (from JSON files)
        else:
            # Handle different possible field names
            name = idea.get('name') or idea.get('title') or 'unnamed_idea'
            title = idea.get('title') or idea.get('name') or 'Untitled'
            description = idea.get('description') or idea.get('content') or ''
            method = idea.get('method') or ''

            return {
                'name': name[:50] if name else 'unnamed_idea',  # Limit name length
                'title': title,
                'description': description,
                'method': method
            }

    def _parse_final_info(self, final_info_path):
        """Read final_info.json and keep only scalar metrics."""
        try:
            with open(final_info_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self.logger.warning(f"Could not read final_info at {final_info_path}: {exc}")
            return {}

        metrics = {}
        for key, value in payload.items():
            if isinstance(value, dict) and "means" in value:
                metrics[key] = value["means"]
            else:
                metrics[key] = value
        return metrics

    def _read_tail(self, file_path, limit=1200):
        """Return tail snippet of a text file for quick diagnostics."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return ""
        if len(content) <= limit:
            return content
        return content[-limit:]

    def _collect_run_history(self, artifact_dir):
        """Collect baseline metrics and run-by-run history for downstream analysis."""
        history = {
            "baseline": {},
            "runs": []
        }
        if not artifact_dir or not osp.exists(artifact_dir):
            return history

        baseline_file = osp.join(artifact_dir, "run_0", "final_info.json")
        if osp.exists(baseline_file):
            history["baseline"] = self._parse_final_info(baseline_file)

        run_dirs = [
            entry for entry in os.listdir(artifact_dir)
            if re.match(r"run_\\d+", entry)
        ]
        for run_dir in sorted(run_dirs, key=lambda x: int(x.split("_")[1])):
            if run_dir == "run_0":
                continue
            run_path = osp.join(artifact_dir, run_dir)
            run_entry = {
                "run_id": run_dir,
                "status": "unknown"
            }
            final_path = osp.join(run_path, "final_info.json")
            if osp.exists(final_path):
                run_entry["status"] = "success"
                run_entry["metrics"] = self._parse_final_info(final_path)
                run_entry["final_info_path"] = final_path
            else:
                run_entry["status"] = "failed"
                traceback_path = osp.join(run_path, "traceback.log")
                if osp.exists(traceback_path):
                    run_entry["traceback"] = self._read_tail(traceback_path)
            history["runs"].append(run_entry)
        return history

    def setup_experiment_folder(self, base_dir, results_dir, idea):
        """Create experiment folder and setup files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        idea_info = self._extract_idea_info(idea)
        idea_name = f"{timestamp}_{idea_info['name']}"
        folder_name = osp.join(results_dir, idea_name)
        
        if osp.exists(folder_name):
            raise FileExistsError(f"Folder already exists: {folder_name}")
        
        shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)

        # Ensure experiment.py exists in the experiment folder
        experiment_src = osp.join(base_dir, "experiment.py")
        experiment_dst = osp.join(folder_name, "experiment.py")
        if not osp.exists(experiment_dst):
            if osp.exists(experiment_src):
                shutil.copy2(experiment_src, experiment_dst)
                print(f"Copied experiment.py from {experiment_src} to {experiment_dst}")
            else:
                raise FileNotFoundError(f"experiment.py not found in base directory: {experiment_src}")

        # Ensure run_0/experiment.py exists (needed for baseline comparison)
        run0_dir = osp.join(folder_name, "run_0")
        run0_experiment = osp.join(run0_dir, "experiment.py")
        if osp.exists(run0_dir) and not osp.exists(run0_experiment):
            if osp.exists(experiment_src):
                shutil.copy2(experiment_src, run0_experiment)
                print(f"Copied baseline experiment.py to {run0_experiment}")
            else:
                print(f"Warning: Could not copy baseline experiment.py to run_0")
        
        # Create notes file
        notes_path = osp.join(folder_name, "notes.txt")
        with open(notes_path, "w") as f:
            f.write(f"# Name: {idea_info['name']}\n")
            f.write(f"# Title: {idea_info['title']}\n")
            f.write(f"# Description: {idea_info['description']}\n")
            f.write(f"# Method: {idea_info['method']}\n")
            f.write(f"## Run 0: Baseline\n")
        
        return folder_name, idea_name
    
    def setup_logging_redirect(self, folder_name):
        """Redirect stderr to log file, keep stdout visible for aider output"""
        log_path = osp.join(folder_name, "log.txt")
        log_file = open(log_path, "a")
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        # Keep stdout visible so we can see aider's output
        # Only redirect stderr to capture errors
        sys.stderr = log_file
        return original_stdout, original_stderr, log_file
    
    def restore_logging(self, original_stdout, original_stderr, log_file):
        """Restore stdout/stderr"""
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
    
    def run_aider_experiment(self, base_dir, results_dir, idea):
        """Run experiment using Aider backend"""
        summary = {
            "success": False,
            "artifact_dir": None,
            "idea_label": None,
            "error": None,
        }

        folder_name, idea_name = self.setup_experiment_folder(base_dir, results_dir, idea)
        summary["artifact_dir"] = folder_name
        summary["idea_label"] = idea_name
        
        # Load baseline results
        baseline_path = osp.join(base_dir, "run_0", "final_info.json")
        if osp.exists(baseline_path):
            with open(baseline_path, "r") as f:
                baseline_results = json.load(f)
        else:
            baseline_results = {}
        
        original_stdout, original_stderr, log_file = self.setup_logging_redirect(folder_name)
        
        try:
            self.logger.info(f"Starting Aider experiment: {idea_name}")
            
            exp_file = osp.join(folder_name, "experiment.py")
            notes_file = osp.join(folder_name, "notes.txt")
            
            io = InputOutput(
                yes=True,
                chat_history_file=osp.join(folder_name, f"{idea_name}_aider.txt")
            )
            # Get experiment model from config file
            experiment_model = (
                self.config.get("experiment", {}).get("model") or  # Config file
                "anthropic/claude-3-7-sonnet-20250219"  # Final fallback
            )
            self.logger.info(f"Using experiment model: {experiment_model}")
            main_model = Model(experiment_model)
            coder = Coder.create(
                main_model=main_model,
                fnames=[exp_file, notes_file],
                io=io,
                stream=True,  # Enable streaming to see aider output
                use_git=False,
                edit_format="diff"
            )
            
            success = perform_experiments_aider(idea, folder_name, coder, baseline_results)
            summary["success"] = bool(success)
            
            self.logger.info(f"Aider experiment {'succeeded' if success else 'failed'}: {idea_name}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Aider experiment error: {str(e)}")
            summary["error"] = str(e)
            return summary
        finally:
            self.restore_logging(original_stdout, original_stderr, log_file)
    
    # def run_openhands_experiment(self, base_dir, results_dir, idea):
    #     """Run experiment using OpenHands backend"""
    #     folder_name, idea_name = self.setup_experiment_folder(base_dir, results_dir, idea)
        
    #     original_stdout, original_stderr, log_file = self.setup_logging_redirect(folder_name)
        
    #     try:
    #         self.logger.info(f"Starting OpenHands experiment: {idea_name}")
            
    #         # Get OpenHands-specific config
    #         openhands_config = self.config.get("experiment", {}).get("openhands", {})
    #         code_server_path = "/workspace"
    #         mount_paths = openhands_config.get("mount_paths", [])
    #         uri_prefix = openhands_config.get("uri_prefix", "ws://localhost:8001/ws/")

    #         success = perform_experiments_openhands(
    #             idea,
    #             folder_name,
    #             code_server_path,
    #             mount_paths,
    #             uri_prefix
    #         )
            
    #         self.logger.info(f"OpenHands experiment {'succeeded' if success else 'failed'}: {idea_name}")
    #         return success
            
    #     except Exception as e:
    #         self.logger.error(f"OpenHands experiment error: {str(e)}")
    #         return False
    #     finally:
    #         self.restore_logging(original_stdout, original_stderr, log_file)
    
    def run_experiments(self, base_dir, results_dir, ideas):
        """Run experiments for all ideas"""
        results = []
        
        for idx, idea in enumerate(ideas, 1):
            idea_info = self._extract_idea_info(idea)
            idea_name = idea_info['name']
            self.logger.info(f"Processing idea {idx}/{len(ideas)}: {idea_name}")
            
            try:
                if self.backend == "aider":
                    execution_result = self.run_aider_experiment(base_dir, results_dir, idea)
                elif self.backend == "openhands":
                    raise NotImplementedError("OpenHands backend is not implemented in this version.")
                    # success = self.run_openhands_experiment(base_dir, results_dir, idea)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                success = execution_result.get("success", False) if self.backend == "aider" else False
                artifact_dir = execution_result.get("artifact_dir") if self.backend == "aider" else None
                run_history = self._collect_run_history(artifact_dir)

                result_entry = {
                    'idea_name': idea_name,
                    'success': success,
                    'artifact_dir': artifact_dir,
                    'run_history': run_history,
                    'idea_details': idea_info,
                }
                if execution_result.get("error"):
                    result_entry['error'] = execution_result['error']
                
                results.append(result_entry)
                
            except Exception as e:
                self.logger.error(f"Failed to run experiment for {idea_name}: {str(e)}")
                results.append({
                    'idea_name': idea_name,
                    'success': False,
                    'error': str(e)
                })
        
        return results
