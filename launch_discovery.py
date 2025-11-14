"""
Launch InternAgent 
"""
import os
import os.path as osp
import sys
import json
import argparse
import asyncio
import logging
import torch
import yaml
import shutil
import copy
from datetime import datetime
from dotenv import load_dotenv

# Import MAS components
from internagent.stage import IdeaGenerator, ExperimentRunner
from internagent.mas.agents.agent_factory import AgentFactory
from internagent.mas.models.model_factory import ModelFactory

load_dotenv()

# ============================================================================
# Logging Configuration
# ============================================================================
def setup_logging():
    """Setup logging configuration"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = osp.join(log_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_internagent.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger("InternAgent")


# ============================================================================
# Argument Parser
# ============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Integrated InternAgent Pipeline: Idea Generation + Experiment Execution"
    )
    
    # ========================================
    # Task Configuration
    # ========================================
    task_group = parser.add_argument_group('Task Configuration')
    task_group.add_argument(
        "--task",
        type=str,
        default="AutoSeg",
        help="Task name or path to task directory. If it's a name, will use tasks/{task}; if it's a path, will use it directly"
    )
    task_group.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results output directory (defaults to results/{task_name})"
    )
    task_group.add_argument(
        "--summary_only",
        action="store_true",
        help="Only run the experiment summary agent using existing artifacts (skips idea generation and experiment execution)"
    )
    task_group.add_argument(
        "--config",
        type=str,
        default='config/default_config.yaml',
        help="Path to configuration file"
    )
    task_group.add_argument(
        "--iterative",
        action="store_true",
        help="Enable iterative mode: repeat experiment + summary cycles until manually stopped"
    )
    task_group.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Optional upper bound on iterations when --iterative is enabled"
    )
    
    # ========================================
    # Idea Generation Phase
    # ========================================
    idea_group = parser.add_argument_group('Idea Generation Phase')
    idea_group.add_argument(
        "--skip_idea_generation",
        action="store_true",
        help="Skip idea generation and use existing ideas from idea_path"
    )
    idea_group.add_argument(
        "--idea_path",
        type=str,
        default=None,
        help="Path to existing ideas JSON (used when skip_idea_generation=True)"
    )
    idea_group.add_argument(
        "--ref_code_path",
        type=str,
        default=None,
        help="Baseline reference code path (defaults to {task_dir}/experiment.py)"
    )
    idea_group.add_argument(
        "--offline_feedback",
        type=str,
        default='config/feedback_global.json',
        help="Offline feedback file for idea generation"
    )
    
    # ========================================
    # Experiment Execution Config
    # ========================================
    exp_group = parser.add_argument_group('Experiment Execution Phase')
    exp_group.add_argument(
        "--exp_backend",
        type=str,
        required=True,
        choices=["aider", "openhands"],
        help="Experiment backend to use"
    )
    # Note: Model configuration is handled through config file (experiment.model)
    exp_group.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g., '0,1,2')"
    )

    # Note: OpenHands-specific configuration (mount_paths, uri_prefix) is handled through config file
    
    return parser.parse_args()


def merge_results(accumulated, new_results, iteration_idx):
    """Merge per-iteration experiment results for long-horizon summaries."""
    if not accumulated:
        accumulated = []
    existing_map = {r.get('idea_name'): r for r in accumulated if r.get('idea_name')}

    def _annotate_history(history, iteration_label):
        history = history or {}
        baseline = history.get('baseline') or {}
        if baseline and 'iteration' not in baseline:
            baseline = dict(baseline)
            baseline['iteration'] = iteration_label
        runs = []
        for run in history.get('runs', []) or []:
            annotated = dict(run)
            annotated.setdefault('iteration', iteration_label)
            runs.append(annotated)
        timeline = []
        for run in history.get('timeline', []) or []:
            annotated = dict(run)
            annotated.setdefault('iteration', iteration_label)
            timeline.append(annotated)
        return {
            'baseline': baseline,
            'runs': runs,
            'timeline': timeline if timeline else runs,
        }

    for entry in new_results or []:
        idea_name = entry.get('idea_name')
        if not idea_name:
            continue
        entry_copy = copy.deepcopy(entry)
        entry_copy.setdefault('iteration_log', [])
        entry_copy['iteration_log'].append({
            'iteration': iteration_idx,
            'success': entry_copy.get('success', False),
            'error': entry_copy.get('error', '')
        })
        annotated_history = _annotate_history(entry_copy.get('run_history'), iteration_idx)
        entry_copy['run_history'] = annotated_history

        if idea_name in existing_map:
            existing = existing_map[idea_name]
            existing['success'] = entry_copy.get('success', existing.get('success'))
            if entry_copy.get('error'):
                existing['error'] = entry_copy.get('error')
            if entry_copy.get('artifact_dir'):
                existing['artifact_dir'] = entry_copy.get('artifact_dir')
            if entry_copy.get('idea_details'):
                existing['idea_details'] = entry_copy.get('idea_details')
            existing.setdefault('iteration_log', [])
            existing['iteration_log'].extend(entry_copy['iteration_log'])

            existing_history = existing.get('run_history', {'baseline': {}, 'runs': [], 'timeline': []})
            if not existing_history.get('baseline') and annotated_history.get('baseline'):
                existing_history['baseline'] = annotated_history['baseline']
            existing_history.setdefault('runs', [])
            existing_history['runs'].extend(annotated_history.get('runs', []))
            existing_history.setdefault('timeline', [])
            existing_history['timeline'].extend(annotated_history.get('timeline', []))
            existing['run_history'] = existing_history
        else:
            accumulated.append(entry_copy)
            existing_map[idea_name] = entry_copy

    return accumulated


def generate_summary_artifacts(
    logger,
    args,
    config,
    results,
    top_ideas,
    session_json,
    active_skip,
    canonical_ideas_path,
    canonical_session_json,
    current_output_dir,
    root_output_dir,
    iteration_label,
):
    try:
        task_meta = {}
        prompt_json_path = osp.join(args.task_dir, "prompt.json")
        if osp.exists(prompt_json_path):
            with open(prompt_json_path, 'r', encoding='utf-8') as pf:
                task_meta = json.load(pf)

        ideas_data = []
        candidate_idea_paths = [osp.join(current_output_dir, "ideas.json")]
        if canonical_ideas_path:
            candidate_idea_paths.append(canonical_ideas_path)
        for idea_path in candidate_idea_paths:
            if idea_path and osp.exists(idea_path):
                try:
                    with open(idea_path, 'r', encoding='utf-8') as inf:
                        loaded_ideas = json.load(inf)
                    if isinstance(loaded_ideas, dict):
                        if 'ideas' in loaded_ideas:
                            ideas_data = loaded_ideas.get('ideas', [])
                        elif 'hypotheses' in loaded_ideas:
                            ideas_data = loaded_ideas.get('hypotheses', [])
                        else:
                            ideas_data = [loaded_ideas]
                    elif isinstance(loaded_ideas, list):
                        ideas_data = loaded_ideas
                    if ideas_data:
                        break
                except Exception:
                    continue
        if not ideas_data:
            try:
                ideas_data = [
                    (i.get('refined_method_details') or i.get('method_details') or i) for i in top_ideas
                ]
            except Exception:
                ideas_data = []

        session_traj = {}
        session_source = None
        if not active_skip and session_json and osp.exists(session_json):
            session_source = session_json
        elif canonical_session_json and osp.exists(canonical_session_json):
            session_source = canonical_session_json
        if session_source:
            try:
                with open(session_source, 'r', encoding='utf-8') as sj:
                    session_traj = json.load(sj)
            except Exception:
                session_traj = {}

        agent_cfg = (config.get('agents', {}).get('experiment_summary', {}) if isinstance(config, dict) else {})
        merged_cfg = dict(agent_cfg)
        merged_cfg['_global_config'] = config if isinstance(config, dict) else {}

        summary_agent = AgentFactory.create_agent(
            agent_type='experiment_summary',
            config=merged_cfg,
            model_factory=ModelFactory()
        )

        context_for_agent = {
            'task': {
                'name': args.task_name,
                'description': task_meta.get('task_description', ''),
                'domain': task_meta.get('domain', ''),
                'constraints': task_meta.get('constraints', []),
                'background': task_meta.get('background', ''),
            },
            'ideas': ideas_data,
            'results': results,
            'session_traj': session_traj,
        }

        agent_output = asyncio.run(summary_agent.execute(context_for_agent, {}))

        iteration_suffix = f"iteration_{iteration_label}" if iteration_label else "manual"
        summary_llm_json = osp.join(
            current_output_dir,
            f'experiment_summary_{iteration_suffix}.llm.json'
        )
        with open(summary_llm_json, 'w', encoding='utf-8') as jf:
            json.dump(agent_output, jf, ensure_ascii=False, indent=2)

        summary_md = osp.join(
            current_output_dir,
            f'experiment_summary_{iteration_suffix}.md'
        )
        with open(summary_md, 'w', encoding='utf-8') as mf:
            mf.write(agent_output.get('summary_markdown', ''))

        aggregate_summary_llm = osp.join(root_output_dir, 'experiment_summary.llm.json')
        with open(aggregate_summary_llm, 'w', encoding='utf-8') as jf:
            json.dump(agent_output, jf, ensure_ascii=False, indent=2)
        aggregate_summary_md = osp.join(root_output_dir, 'experiment_summary.md')
        with open(aggregate_summary_md, 'w', encoding='utf-8') as mf:
            mf.write(agent_output.get('summary_markdown', ''))

        logger.info(f"Summary artifacts saved to {summary_llm_json} and {summary_md}")
    except Exception as e:
        logger.warning(f"Experiment summary agent failed: {e}")


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    logger = setup_logging()
    args = parse_arguments()

    if '/' in args.task or '\\' in args.task or osp.isdir(args.task):
        args.task_dir = args.task
        args.task_name = osp.basename(args.task.rstrip('/\\'))
    else:
        args.task_dir = osp.join("tasks", args.task)
        args.task_name = args.task

    if not osp.exists(args.task_dir):
        raise FileNotFoundError(f"Task directory not found: {args.task_dir}")

    if args.ref_code_path is None:
        args.ref_code_path = osp.join(args.task_dir, "experiment.py")

    if args.output_dir is None:
        args.output_dir = osp.join("results", args.task_name)
    else:
        args.output_dir = osp.join("results", args.output_dir)

    root_output_dir = args.output_dir
    os.makedirs(root_output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("InternAgent Pipeline Started")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Task Directory: {args.task_dir}")
    logger.info(f"Experiment Backend: {args.exp_backend}")
    logger.info(f"Output Directory: {root_output_dir}")
    if args.iterative:
        logger.info("Iterative mode: enabled")
    logger.info("=" * 80)

    config = {}
    if args.config and osp.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            logger.info(f"Loaded experiment config from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load experiment config: {e}")

    summary_path = osp.join(root_output_dir, "discovery_summary.json")

    if args.summary_only:
        if not osp.exists(summary_path):
            raise FileNotFoundError(f"Discovery summary not found at {summary_path}")
        with open(summary_path, 'r', encoding='utf-8') as sf:
            prior_summary = json.load(sf)
        logger.info(f"Loaded existing summary from {summary_path}")
        results = prior_summary.get('results', [])
        canonical_ideas_path = args.idea_path or osp.join(root_output_dir, "ideas_canonical.json")
        generate_summary_artifacts(
            logger,
            args,
            config,
            results,
            top_ideas=[],
            session_json=None,
            active_skip=True,
            canonical_ideas_path=canonical_ideas_path,
            canonical_session_json=None,
            current_output_dir=root_output_dir,
            root_output_dir=root_output_dir,
            iteration_label="manual",
        )
        return

    if args.exp_backend == "openhands":
        openhands_config = config.get("experiment", {}).get("openhands", {})
        mount_paths = openhands_config.get("mount_paths", [])
        uri_prefix = openhands_config.get("uri_prefix", "ws://localhost:8001/ws/")
        if not mount_paths:
            logger.warning("No mount paths specified in config for OpenHands backend")
        else:
            logger.info(f"OpenHands mount paths: {mount_paths}")
        logger.info(f"OpenHands URI prefix: {uri_prefix}")

    if args.gpus:
        gpu_ids = [int(gid) for gid in args.gpus.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"Using GPUs: {gpu_ids}")
    else:
        available_gpus = list(range(torch.cuda.device_count()))
        logger.info(f"Using all available GPUs: {available_gpus}")

    aggregated_results = []
    iteration = 1
    canonical_ideas_path = args.idea_path if args.skip_idea_generation and args.idea_path else None
    canonical_session_json = None
    base_skip_flag = args.skip_idea_generation

    while True:
        if args.iterative:
            current_output_dir = osp.join(root_output_dir, f"iteration_{iteration}")
        else:
            current_output_dir = root_output_dir
        args.output_dir = current_output_dir
        os.makedirs(current_output_dir, exist_ok=True)

        logger.info("=" * 80)
        logger.info(f"Iteration {iteration} started")
        logger.info("=" * 80)

        active_skip = base_skip_flag or (args.iterative and iteration > 1)
        args.skip_idea_generation = active_skip

        session_json = None
        top_ideas = []

        if active_skip:
            idea_source = args.idea_path if base_skip_flag and iteration == 1 else canonical_ideas_path
            if not idea_source or not osp.exists(idea_source):
                raise FileNotFoundError(
                    "Idea path not found for iterative run; ensure ideas.json exists from a prior iteration."
                )
            with open(idea_source, 'r', encoding='utf-8') as f:
                ideas_data = json.load(f)
            if isinstance(ideas_data, dict) and 'hypotheses' in ideas_data and 'top_hypotheses' in ideas_data:
                top_ideas = [item for item in ideas_data['hypotheses'] if item['id'] in ideas_data['top_hypotheses']]
            elif isinstance(ideas_data, list):
                top_ideas = ideas_data
            else:
                top_ideas = ideas_data.get('ideas', []) if isinstance(ideas_data, dict) else []
            logger.info(f"Loaded {len(top_ideas)} ideas from {idea_source}")
            session_json = canonical_session_json or idea_source
        else:
            logger.info("Starting idea generation with MAS...")
            idea_generator = IdeaGenerator(args, logger)
            try:
                top_ideas, session_json = asyncio.run(idea_generator.generate_ideas())
            except Exception as e:
                logger.error(f"Idea generation failed: {str(e)}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            ideas_output = osp.join(current_output_dir, "ideas.json")
            os.makedirs(osp.dirname(ideas_output), exist_ok=True)
            aligned_ideas = [idea['refined_method_details'] for idea in top_ideas]
            with open(ideas_output, 'w') as f:
                json.dump(aligned_ideas, f, indent=4)
            logger.info(f"Ideas saved to {ideas_output}")

            if args.iterative:
                canonical_ideas_path = canonical_ideas_path or osp.join(root_output_dir, "ideas_canonical.json")
                shutil.copy2(ideas_output, canonical_ideas_path)
            if not canonical_session_json and session_json:
                canonical_session_json = session_json

        logger.info(f"Starting experiment execution with {args.exp_backend} backend")
        logger.info(f"Number of ideas to test: {len(top_ideas)}")

        experiment_runner = ExperimentRunner(args, logger, config)
        try:
            iteration_results = experiment_runner.run_experiments(
                base_dir=args.task_dir,
                results_dir=current_output_dir,
                ideas=top_ideas
            )
        except Exception as e:
            logger.error(f"Experiment execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        aggregated_results = merge_results(aggregated_results, iteration_results, iteration) if args.iterative else iteration_results
        summary_results_input = aggregated_results if args.iterative else iteration_results

        iteration_successful = sum(1 for r in iteration_results if r.get('success'))
        iteration_failed = len(iteration_results) - iteration_successful
        logger.info(f"[Iteration {iteration}] Successful: {iteration_successful} | Failed: {iteration_failed}")

        total_successful = sum(1 for r in summary_results_input if r.get('success'))
        total_failed = len(summary_results_input) - total_successful
        logger.info(f"[Cumulative] Successful: {total_successful} | Failed: {total_failed}")

        logger.info("\nDetailed Results (current iteration):")
        for idx, result in enumerate(iteration_results, 1):
            status = "✓ SUCCESS" if result.get('success') else "✗ FAILED"
            logger.info(f"  {idx}. {result.get('idea_name')}: {status}")
            if result.get('error'):
                logger.info(f"     Error: {result['error']}")

        iteration_summary = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'task': args.task_name,
            'task_dir': args.task_dir,
            'exp_backend': args.exp_backend,
            'output_dir': current_output_dir,
            'ideas_source': session_json if not active_skip else (args.idea_path or canonical_ideas_path),
            'skip_idea_generation': active_skip,
            'model': (
                config.get("experiment", {}).get("model") or
                "anthropic/claude-3-7-sonnet-20250219"
            ),
            'total_ideas': len(iteration_results),
            'successful': iteration_successful,
            'failed': iteration_failed,
            'results': iteration_results
        }

        iteration_summary_path = osp.join(current_output_dir, "discovery_summary.json")
        with open(iteration_summary_path, 'w') as f:
            json.dump(iteration_summary, f, indent=4)
        logger.info(f"Iteration summary saved to {iteration_summary_path}")

        aggregate_summary = dict(iteration_summary)
        aggregate_summary['results'] = summary_results_input
        aggregate_summary['successful'] = total_successful
        aggregate_summary['failed'] = total_failed
        aggregate_summary['total_ideas'] = len(summary_results_input)
        aggregate_summary['output_dir'] = root_output_dir
        with open(summary_path, 'w') as f:
            json.dump(aggregate_summary, f, indent=4)
        logger.info(f"Cumulative summary updated at {summary_path}")

        generate_summary_artifacts(
            logger,
            args,
            config,
            summary_results_input,
            top_ideas,
            session_json,
            active_skip,
            canonical_ideas_path,
            canonical_session_json,
            current_output_dir,
            root_output_dir,
            iteration,
        )

        if not args.iterative:
            break

        if args.max_iterations and iteration >= args.max_iterations:
            logger.info(f"Reached max iterations ({args.max_iterations}). Stopping.")
            break

        user_input = ""
        if sys.stdin and sys.stdin.isatty():
            try:
                user_input = input("Proceed to next iteration? (Y/n): ").strip().lower()
            except EOFError:
                user_input = ""
        if user_input in ("n", "no", "stop", "q"):
            logger.info("Stopping iterative loop per user request.")
            break

        iteration += 1
        if canonical_ideas_path:
            args.idea_path = canonical_ideas_path

    logger.info("=" * 80)
    logger.info("All done!")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDiscovery pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
