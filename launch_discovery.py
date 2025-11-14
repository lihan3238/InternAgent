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
from datetime import datetime
from dotenv import load_dotenv

# Import MAS components
from internagent.stage import IdeaGenerator, ExperimentRunner

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
        "--config",
        type=str,
        default='config/default_config.yaml',
        help="Path to configuration file"
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


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    logger = setup_logging()
    args = parse_arguments()
    
    # Setup task directory
    # If task is a path (contains / or \), use it directly; otherwise use tasks/{task}
    if '/' in args.task or '\\' in args.task or osp.isdir(args.task):
        args.task_dir = args.task
        args.task_name = osp.basename(args.task.rstrip('/\\'))
    else:
        args.task_dir = osp.join("tasks", args.task)
        args.task_name = args.task
    
    if not osp.exists(args.task_dir):
        raise FileNotFoundError(f"Task directory not found: {args.task_dir}")
    
    # Setup reference code path
    if args.ref_code_path is None:
        args.ref_code_path = osp.join(args.task_dir, "experiment.py")
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = osp.join("results", args.task_name)
    else:
        args.output_dir = osp.join("results", args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("InternAgent Pipeline Started")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Task Directory: {args.task_dir}")
    logger.info(f"Experiment Backend: {args.exp_backend}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Step 1: Idea Generation
    if args.skip_idea_generation:
        logger.info("Skipping idea generation, loading existing ideas...")
        if not args.idea_path or not osp.exists(args.idea_path):
            raise FileNotFoundError(f"Idea path not found: {args.idea_path}")
        
        with open(args.idea_path, 'r') as f:
            ideas_data = json.load(f)
        
        # Extract top hypotheses
        if 'hypotheses' in ideas_data and 'top_hypotheses' in ideas_data:
            top_ideas = [
                item for item in ideas_data['hypotheses']
                if item['id'] in ideas_data['top_hypotheses']
            ]
        else:
            # Assume it's already a list of ideas
            top_ideas = ideas_data
        
        logger.info(f"Loaded {len(top_ideas)} ideas from {args.idea_path}")
        session_json = args.idea_path
        
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
        
        # Save ideas in standard format
        ideas_output = osp.join(args.output_dir, "ideas.json")
        os.makedirs(osp.dirname(ideas_output), exist_ok=True)
        
        aligned_ideas = [idea['refined_method_details'] for idea in top_ideas]
        with open(ideas_output, 'w') as f:
            json.dump(aligned_ideas, f, indent=4)
        
        logger.info(f"Ideas saved to {ideas_output}")
    
    # Step 2: Experiment Execution
    logger.info("=" * 80)
    logger.info(f"Starting experiment execution with {args.exp_backend} backend")
    logger.info(f"Number of ideas to test: {len(top_ideas)}")
    logger.info("=" * 80)
    
    # Setup GPU configuration
    if args.gpus:
        gpu_ids = [int(gid) for gid in args.gpus.split(',')]
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"Using GPUs: {gpu_ids}")
    else:
        available_gpus = list(range(torch.cuda.device_count()))
        logger.info(f"Using all available GPUs: {available_gpus}")
    
    # Load config for experiment runner
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

    # Validate backend-specific requirements
    if args.exp_backend == "openhands":
        openhands_config = config.get("experiment", {}).get("openhands", {})
        mount_paths = openhands_config.get("mount_paths", [])
        uri_prefix = openhands_config.get("uri_prefix", "ws://localhost:8001/ws/")

        if not mount_paths:
            logger.warning("No mount paths specified in config for OpenHands backend")
        else:
            logger.info(f"OpenHands mount paths: {mount_paths}")
        logger.info(f"OpenHands URI prefix: {uri_prefix}")

    # Run experiments
    experiment_runner = ExperimentRunner(args, logger, config)
    
    try:
        results = experiment_runner.run_experiments(
            base_dir=args.task_dir,
            results_dir=args.output_dir,
            ideas=top_ideas
        )
    except Exception as e:
        logger.error(f"Experiment execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Summary
    logger.info("=" * 80)
    logger.info("InternAgent Run Completed")
    logger.info("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    logger.info(f"Total ideas tested: {len(results)}")
    logger.info(f"Successful experiments: {successful}")
    logger.info(f"Failed experiments: {failed}")
    
    # Print detailed results
    logger.info("\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        logger.info(f"  {i}. {result['idea_name']}: {status}")
        if 'error' in result:
            logger.info(f"     Error: {result['error']}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'task': args.task_name,
        'task_dir': args.task_dir,
        'exp_backend': args.exp_backend,
        'output_dir': args.output_dir,
        'ideas_source': session_json if not args.skip_idea_generation else args.idea_path,
        'skip_idea_generation': args.skip_idea_generation,
        'model': (
            config.get("experiment", {}).get("model") or  # Config file
            "anthropic/claude-3-7-sonnet-20250219"  # Final fallback
        ),
        'total_ideas': len(results),
        'successful': successful,
        'failed': failed,
        'results': results
    }
    
    summary_path = osp.join(args.output_dir, "discovery_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"\nSummary saved to {summary_path}")
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
