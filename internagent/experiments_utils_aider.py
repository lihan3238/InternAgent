import shutil
import os.path as osp
import subprocess
from subprocess import TimeoutExpired
import sys
import json
import re
import os

from internagent.prompts import (
    CODER_PROMPT_AIDER, 
    NEXT_EXPERIMENT_PROMPT, 
    CODE_STRUCTURE_PROMPT, 
    DEBUG_PROMPT_WITH_STRUCTURE
    )

import filecmp

MAX_ITERS = 5
MAX_RUNS = 5
MAX_STDERR_OUTPUT = 3000


def extract_idea_info(idea):
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


# return (file, line, function, content), message
def info_traceback(stderr):
    pattern = r'File "(.*)", line (\d+), in (.+)\n (.*)'
    matches = re.findall(pattern, stderr)
    match = re.search(rf'\w*Error\w*(.*)', stderr, re.DOTALL)
    message = match.group(1).strip()
    externel = []
    for match in matches:
        if match[0].split('/')[-1] == 'experiment.py':
            continue
        else:
            externel.append(match)
    for e in externel:
        matches.remove(e)

    return matches, message


# RUN EXPERIMENT
def run_experiment(folder_name, run_num, timeout=27000):
    cwd = osp.abspath(folder_name)

    # Check if experiment.py exists before trying to copy
    experiment_src = osp.join(cwd, "experiment.py")
    if not osp.exists(experiment_src):
        raise FileNotFoundError(f"experiment.py not found in experiment directory: {experiment_src}")

    # COPY CODE SO WE CAN SEE IT.
    run_dir = osp.join(cwd, f"run_{run_num}")
    if not osp.exists(run_dir):
        os.mkdir(run_dir)

    experiment_dst = osp.join(run_dir, "experiment.py")
    shutil.copy(experiment_src, experiment_dst)

    
    # Copy plot.py if it exists in the experiment folder
    # plot_src = osp.join(cwd, "plot.py")
    # if osp.exists(plot_src):
    #     plot_dst = osp.join(run_dir, "plot.py")
    #     shutil.copy(plot_src, plot_dst)
    #     print(f"[INFO] Copied plot.py to {plot_dst}")plot复制功能文件
    plot_dst = osp.join(run_dir, "plot.py")
    shutil.copy(plot_dst, plot_dst)
    print(f"[INFO] Copied plot.py to {plot_dst}")

    # LAUNCH COMMAND
    command = ["bash", "launcher.sh", f"run_{run_num}"]
    try:
        result = subprocess.run(
            command, cwd=cwd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, timeout=timeout
        )

        if os.path.exists(osp.join(cwd, f"run_{run_num}", "final_info.json")):
            # Experiment succeeded, now run plot.py if it exists绘图部分修正
            plot_py_path = osp.join(cwd, "plot.py")
            run_plot_py_path = osp.join(cwd, f"run_{run_num}", "plot.py")
            if osp.exists(plot_py_path) or osp.exists(run_plot_py_path):
                plot_script = run_plot_py_path if osp.exists(run_plot_py_path) else plot_py_path
                print(f"[INFO] Running visualization with {plot_script}...")
                plot_result = subprocess.run(
                    ["python", plot_script, "--out_dir", f"run_{run_num}"],
                    cwd=cwd,
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    timeout=300  # 5 minute timeout for visualization
                )
                if plot_result.returncode == 0:
                    print(f"[INFO] Visualization completed successfully")
                else:
                    print(f"[WARNING] Visualization failed (return code {plot_result.returncode}), but experiment succeeded")
                    if plot_result.stderr:
                        print(f"[WARNING] Plot error: {plot_result.stderr[:500]}")  # Print first 500 chars
            # 至此
            results = {}

            baseline_path = osp.join(cwd, "run_0", "final_info.json")
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    baseline_data = json.load(f)
                baseline_results = {k: v["means"] for k, v in baseline_data.items()}
                results["baseline"] = baseline_results

            for run_idx in range(1, run_num + 1):
                run_path = osp.join(cwd, f"run_{run_idx}", "final_info.json")
                if os.path.exists(run_path):
                    with open(run_path, "r") as f:
                        run_data = json.load(f)
                    run_results = {k: v["means"] for k, v in run_data.items()}
                    results[f"improve_{run_idx}"] = run_results
                    # with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                    #     results = json.load(f)
                    #     results = {k: v["means"] for k, v in results.items()}

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)
            traceback, message, tb = None, None, None
            return result.returncode, next_prompt, traceback, message

        if result.stderr:
            print(result.stderr, file=sys.stderr)
            traceback_file = osp.join(cwd, f"run_{run_num}", "traceback.log")
            if osp.exists(traceback_file):
                with open(traceback_file, "r") as file:
                    tb = file.read()
                traceback, message = info_traceback(tb)
            else:
                # Use stderr as fallback if traceback.log doesn't exist
                tb = result.stderr
                traceback, message = info_traceback(tb) if tb else (None, None)
        else:
            traceback, message, tb = None, None, None

        if result.returncode != 0:
            print(f"Run {run_num} failed with return code {result.returncode}")
            if osp.exists(osp.join(cwd, f"run_{run_num}")):
                shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
            print(f"Run failed with the following error {result.stderr}")
            if tb:
                stderr_output = tb
            else:
                stderr_output = result.stderr
            if len(stderr_output) > MAX_STDERR_OUTPUT:
                stderr_output = "..." + stderr_output[-MAX_STDERR_OUTPUT:]
            next_prompt = f"Run failed with the following error {stderr_output}"
        else:
            with open(osp.join(cwd, f"run_{run_num}", "final_info.json"), "r") as f:
                results = json.load(f)
            results = {k: v["means"] for k, v in results.items()}

            next_prompt = NEXT_EXPERIMENT_PROMPT.format(RUN_NUM=run_num, RESULTS=results, NEXT_RUN_NUM=run_num+1)

        return result.returncode, next_prompt, traceback, message
    except TimeoutExpired:
        print(f"Run {run_num} timed out after {timeout} seconds")
        if osp.exists(osp.join(cwd, f"run_{run_num}")):
            shutil.rmtree(osp.join(cwd, f"run_{run_num}"))
        next_prompt = f"Run timed out after {timeout} seconds"
        return 1, next_prompt, None, None


# PERFORM EXPERIMENTS
def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    ## RUN EXPERIMENT
    current_iter = 0
    run = 1
    idea_info = extract_idea_info(idea)
    next_prompt = CODER_PROMPT_AIDER.format(
        title=idea_info["title"],
        method=idea_info["method"],
        idea=idea_info["description"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )
    while run < MAX_RUNS + 1:
        if current_iter >= MAX_ITERS:
            print("Max iterations reached")
            break
        coder_out = coder.run(next_prompt) # 1. method2code
        print(coder_out)
        if "litellm.BadRequestError" in coder_out:
            return False
        if "ALL_COMPLETED" in coder_out:
            break
        if filecmp.cmp(os.path.join(folder_name, 'experiment.py'), os.path.join(folder_name, 'run_0', 'experiment.py')):
            print("do not modify code")
            continue
        
        # 2.autodebug
        return_code, next_prompt, traceback, message = run_experiment(folder_name, run)#  request
        # add traceback and code_structure
        if traceback:
            functions_codes = ""
            for t in traceback:
                functions_codes = functions_codes + f"line: {t[1]}, function: {t[2]}, codes: {t[3]} \n"

            code_structure = coder.run(CODE_STRUCTURE_PROMPT.format(error_messages=next_prompt, function_code=functions_codes))

            next_prompt = DEBUG_PROMPT_WITH_STRUCTURE.format(error_messages=next_prompt, code_structure=code_structure)

        if return_code == 0:
            run += 1
            current_iter = 0
        current_iter += 1
    if current_iter >= MAX_ITERS:
        print("Not all experiments completed.")
        return False
    return True
