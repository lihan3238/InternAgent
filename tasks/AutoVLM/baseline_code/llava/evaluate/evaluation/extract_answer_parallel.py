

import os
import re
import time
import argparse

from tqdm import tqdm

import sys
sys.path.append('../')
from llava.evaluate.evaluation.utilities import *

# OpenAI
# from openai import OpenAI
# load demo prompt
from prompts.ext_ans import demo_prompt
from multiprocessing import Pool, cpu_count, Manager
import pdb

def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']

    if response == "":
        return ""
    
    if question_type == 'multi_choice' and response in choices:
        return response
    
    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:
        print("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = get_chat_response(full_prompt)
        return extraction
    except Exception as e:
        print(e)
        print(f"Error in extracting answer for {pid}")

    return ""

def parallel_extract(cur_test_id):
    global shared_save_results
    problem = shared_save_results[cur_test_id]
    assert 'response' in problem
    response = problem['response']    
    extraction  = extract_answer(response, problem)
    shared_save_results[cur_test_id]['extraction'] = extraction
    # shared_save_results[str(cur_test_id)]["hello"]=cur_test_id
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--output_dir', type=str, default='../results')
    parser.add_argument('--output_file', type=str, default='answer.json')
    # model
    # output
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')
    args = parser.parse_args()

    # args
    label = "reaponse"
    result_file = os.path.join(args.output_dir, args.output_file)
    output_file = result_file

    # read results
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    # full pids
    full_pids = list(results.keys())

    # test pids

    test_pids = []
    for pid in full_pids:
        # print(pid)
        if 'extraction' not in results[pid] or not verify_extraction(results[pid]['extraction']):
            test_pids.append(pid)
    
    test_num = len(test_pids)
    # print(test_pids)

    num_group = test_num//args.save_every +1
    manager = Manager()
    shared_save_results = manager.dict({k: manager.dict(v) for k, v in results.items()})  # 创建一个共享的列表

    for i in tqdm(range(num_group)):
        cur_test_pids = test_pids[i*args.save_every:(1+i)*args.save_every]
        parallel_inputs = cur_test_pids

        with Pool(processes=32) as pool:
            list(tqdm(pool.imap(parallel_extract, parallel_inputs), total=len(parallel_inputs), desc = f"Extracting Group {i}."))

        print(f"Saving results to {output_file}...")
        save_results = {k:dict(v) for k, v in shared_save_results.items()}
        save_json(dict(save_results), output_file)
        print(f"Results saved.")
        
    print(f"Saving results to {output_file}...")
    save_results = {k:dict(v) for k, v in shared_save_results.items()}
    save_json(dict(save_results), output_file)
    print(f"Results saved.")

