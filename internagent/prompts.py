CODER_PROMPT_AIDER = """Your goal is to implement the following idea: {idea}
The proposed method is as follows: {method}.
You are given a total of up to {max_runs} runs to complete the necessary experiments. You do not need to use all {max_runs}.
**Note**: It is highly recommended that you implement the core functionality of your code within the first one or two runs, with subsequent experiments focused on performance tuning and hyperparameter optimization based on observed results.

First, plan the list of experiments you would like to run. For example, if you are sweeping over a specific hyperparameter, plan each value you would like to test for each run.

Note that we already provide the vanilla baseline results, so you do not need to re-run it.

For reference, the baseline results are as follows:

{baseline_results}

Then, you need to implement code based on your plan. After you complete each change, we will run the command `python experiment.py --out_dir=run_i' where i is the run number and evaluate the results.
YOUR PROPOSED CHANGE MUST USE THIS COMMAND FORMAT, DO NOT ADD ADDITIONAL COMMAND LINE ARGS.
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
