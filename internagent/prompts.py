CODER_PROMPT_AIDER = """Your goal is to implement the following idea: {idea}
The proposed method is as follows: {method}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.
**Note**: It is highly recommended that you implement the core functionality of your code within the first one or two runs, with subsequent experiments focused on performance tuning and hyperparameter optimization based on observed results.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

Then, you need to implement code based on your plan. After you complete each change, we will run the command `bash launcher.sh run_i` where i is the run number. This command will:
1. Run `experiment.py` to execute the experiment
2. Automatically run `plot.py` to generate visualizations if the experiment succeeds

**Important**: You can modify both `experiment.py` and `plot.py` files:
- `experiment.py`: Contains the main experiment logic, model training, and evaluation
- `plot.py`: Contains visualization code that reads results from `experiment.py` and generates plots
- Both files are available for editing, and `plot.py` will be automatically executed after successful experiments

YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT (`bash launcher.sh run_i`), DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
You can then implement the next thing on your list.

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically:  Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.
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

Previously, you have analyzed the error-related code structure as follows:

{code_structure}

You need to first analyze the error message and list all the possible reasons and code modification plan of the error. Then, modify the code based on the plan. You can refer to the code structure obtained from the previous analysis. 

Any modifications to `argparse` parameters (new/updated) **must enforce the improved implementation as the default behavior** unless explicitly designed as optional. Specifically:  Set `default=<revised_value>` for all altered arguments to ensure the enhanced logic activates automatically without CLI flags. Ensure the improved functionality should be the default experience without requiring users to specify additional command-line parameters.
"""

NEXT_EXPERIMENT_PROMPT = """Run {RUN_NUM} completed. Here are the results:
{RESULTS}

Based on these results:
1. Analyze what worked and what didn't work in your approach.
2. Compare the current run with previous runs and baseline.
3. Decide if you need to re-plan your experiments or continue with your current strategy.
4. If continuing, implement the next improvement on your list.
5. If re-planning, explain why and outline your new approach.

We will run the command `bash launcher.sh {NEXT_RUN_NUM}` to execute your next experiment.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.

If you believe you have completed all necessary experiments and found the optimal solution, respond with 'ALL_COMPLETED'.
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
