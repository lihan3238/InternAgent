CODER_PROMPT_AIDER = """Your goal is to implement the following idea into `experiment.py`.
Idea: {idea}
Method Details: {method}

**Constraints:**
- You have {max_runs} runs. The code must be runnable via `bash launcher.sh run_x`.
- Modify `experiment.py` (and `plot.py` if present). Keep file structure valid.
- Set any argparse changes as defaults so the improved path runs automatically.

**Strategy:**
1) Read current code in this folder.
2) Implement the needed model, data, and training updates.
3) Keep the edits concise; output only the code changes (diffs), no long explanations.

**Baseline Results:**
{baseline_results}
"""

CODER_PROMPT_OPENHANDS = """Your goal is to implement the following idea: {idea_description} in the codes {code_server_path}. 
Please read the code of all the files of {code_server_path} first (important), each time before modifying the file you need to determine the location of the insertion again, and after modification to confirm that the content and location of the modification is correct through observation. After that, I will give you a method and you need to adapt that method appropriately based on the existing baseline code.

## Requirements:
    1. Integrate the core concepts of my improved method into the baseline code
    2. Make necessary adaptations to ensure compatibility with the existing codebase
    3. When conflicts arise between the improved method and baseline implementation: 
        1) Prioritize maintaining the stability of the baseline code 
        2) Adapt the improved method's concepts rather than forcing exact implementation 
        3) Preserve the overall architecture of the baseline while enhancing its functionality
    4. Ensure that the final file to be executed is {code_server_path}/launcher.sh
    5. DO NOT make changes to the original content in the {code_server_path}/launcher.sh, such as the GPU ID, data_root, etc. However, it is allowed to add or modify model-related parameters.
    6. DO NOT attempt to install the environment in the {code_server_path}/launcher.sh
    7. When checking the correctness of the code, ignore the runtime environment issues.

The proposed method is as follows: {method}.

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically:  Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.

"""

CODE_STRUCTURE_PROMPT = """You are an expert code analyst specializing in error detection, debugging, and error handling patterns. Your task is to thoroughly analyze the provided code with a focus on potential errors below:

{error_messages}

You need to focus on error-related aspects of code and analyze their relations. The following functions and codes may highly related to the error which is extracted from the traceback.

{function_code}

Note that you do not need to modify the code in this step and just need to give the error-related code structure.
"""

DEBUG_PROMPT_WITH_STRUCTURE = """You are an expert code debugger specializing in structural analysis and error diagnosis. Your task is to debug the code based on the following error message:

{error_messages}

{code_structure}

Please analyze the error and modify `experiment.py` (and `plot.py` if relevant) to fix it.
- Focus on the specific error above; keep changes targeted.
- Preserve valid previous changes; do not revert working code.
- Ensure argparse defaults activate the improved logic automatically.
"""

NEXT_EXPERIMENT_PROMPT = """Run {RUN_NUM} completed. Here are the results:
{RESULTS}

Based on these results:
1. If the run failed due to incomplete code (e.g., 'NameError', 'ImportError'), continue implementing the missing parts (e.g., Data Loading or Training Loop) that you didn't finish in the previous step.
2. If the run succeeded, analyze the results and compare with baseline.
3. Decide if you need to re-plan or tune hyperparameters.

**REMINDER**: If you need to write a lot of code, split it into smaller chunks to avoid token limits.

We will run the command `bash launcher.sh {NEXT_RUN_NUM}` to execute your next experiment.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
"""


EXPERIMENT_SUMMARY_PROMPT = """
You are the “Experiment Results Analysis Agent”. Use the context below to craft a rigorous, decision-ready summary and plan.

【Context JSON】
{CONTEXT_JSON}

Produce a JSON object that satisfies the provided schema and the following quality bar:
1) `idea_analyses`: for every entry in `experiment_runs`, craft a standalone section with:
   - `idea_name`
   - `plan_brief`: 2 sentences describing intended method/baseline comparison plus one explicit weakness/limitation that surfaced (e.g., capacity gap of HDPMN-SLS).
   - `run_analyses`: using `run_history.timeline`, provide one object per run (baseline + each run_id) covering variables touched, metrics vs. baseline, and success/failure diagnosis.
   - `idea_takeaways`: 2–3 sentences synthesizing what was learned for this idea, emphasizing both strengths and weak points.
   - `idea_next_steps`: 2–4 concrete follow-ups scoped to this idea.
2) `core_idea`: distill the main experimental insight in a single, punchy sentence that highlights what changed vs. baseline.
3) `failure_overview`: 2 sentences synthesizing why the major failures happened (e.g., attention–GNN fusion instability, shallow hierarchy definitions).
4) `failures_and_causes`: 3–5 bullet-style strings of the form “Failure — root cause / evidence”.
5) `result_analysis`: 2–3 compact paragraphs covering (a) what worked, (b) what regressed, and (c) the likely causes, explicitly referencing idea/run names when possible.
6) `literature_links`: cite up to 3 relevant papers, blog posts, or prior work (title or URL). If none, return an empty list.
7) `next_plan`: 
   - `objective`: one crisp outcome statement.
   - `milestones`: 3–5 deliverables expressed as “Milestone — success signal”.
   - `experiments`: list the next key experiments; each entry should mention the variable(s), control/baseline, and expected observation.
   - `metrics`: name at least three concrete metrics (e.g., RMSE, MAE, SMAPE) with target direction or thresholds.
   - `risks`: enumerate the top risks plus explicit mitigation ideas.
8) `improvement_directions`: provide 3–6 actionable, testable directives (“Do X in order to Y; validate via Z”).
9) `summary_markdown`: an English Markdown report with sections titled **Core Idea**, **Per-Idea Highlights**, **Failure Summary**, **Next Plan & Metrics**, and **Risks & Mitigations** so it reads like a research plan.

Return only the JSON response that conforms to the schema.
"""
