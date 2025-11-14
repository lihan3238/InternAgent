import argparse
import io
import logging
import os
import sys
sys.path.append('../')
from datasets import load_dataset
from rich.logging import RichHandler
from tqdm import tqdm

from llava.evaluate.evaluation.build_query import create_query_data
from llava.evaluate.evaluation.utilities import read_json, save_json
import pdb

from llava.evaluate.predict import Predictor

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--dataset_name', type=str, default='AI4Math/MathVista')
    # output
    parser.add_argument('--output_dir', type=str, default='../results/bard')
    parser.add_argument('--output_file', type=str, default='output_bard.json')
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')
    # Local Model
    parser.add_argument('--model_path', type=str, default=None)
   
    args = parser.parse_args()
    return args


def main():
    logging.info("MathVista: Generating Responses - Start")
    args = parse_args()

    # load data
    logging.info(f"Loading dataset {args.dataset_name}, split 'testmini'")
    data_list = load_dataset(args.dataset_name, split='testmini')
    data = {item['pid']: item for item in data_list}

    selected_contexts = ["geometry diagram", "function plot"]
    data = {k: v for k, v in data.items() if v['metadata']['context'] in selected_contexts}

    # load or create query data
    
    logging.info("Creating new query...")

    caption_data = {}
    ocr_data = {}
    query_data = create_query_data(data, caption_data, ocr_data, args)

    # If we were given a custom model path, load that model, otherwise use a remote service model
    model = Predictor(args.model_path)
    logging.info(f"Loading model from {args.model_path}...")
    
    full_pids = list(data.keys())

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file_path):
        logging.info("Results already exist.")
        logging.info(f"Reading {output_file_path}...")
        results = read_json(output_file_path)
    else:
        results = {}


    for i, problem_id in enumerate(tqdm(full_pids)):
        problem: dict = data[problem_id].copy()

        # Remove decoded Image for JSON deserialization
        problem_decoded_image = problem['decoded_image']
        problem.pop('decoded_image')

        query = query_data[problem_id]
        
        logging.debug("--------------------------------------------------------------")
        logging.debug(f"Generating response for problem: {problem_id}...")
        try:
            response = model.predict(prompt=query, image=problem_decoded_image)
            results[problem_id] = problem
            results[problem_id]['query'] = query
            results[problem_id]['response'] = response
            
            logging.debug(f"Query: \n{query}")
            logging.debug(f"Response: \n{response}")
        except Exception as e:
            logging.error(f"Error in extracting answer for {problem_id}")
            logging.error(e)
            results[problem_id] = problem
            results[problem_id]['error'] = str(e)

        if (i % args.save_every == 0 and i > 0) or i == len(full_pids) - 1:
            try:
                save_json(results, output_file_path)
                logging.info(f"Saved results to {output_file_path}")
            except Exception as e:
                logging.info(f"Error in saving {output_file_path}")
                logging.info(e)

    logging.info("MathVista: Generating Responses - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
