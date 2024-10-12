import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from infer.ht_chain.leetcode_submit_sh import add_full_qv, find_code_snippet

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
from pathlib import Path

import typer
from bs4 import BeautifulSoup
import html
from tqdm import tqdm
from ht_chain.utils.json_util import jload
from ht_chain.calculate_metric import calculate_metric_show
from ht_chain.filter_question import filter_questions
from ht_chain.langchain_call import get_langchain_completion
from ht_chain.utils.csv_util import write_data_to_csv
from ht_chain.utils.java_gateway_util import init_gateway
from ht_chain.utils.json_util import read_question_view_json_file, write_question_view_json_file
from ht_chain.utils.logger_util import get_logger, color_message, setup_logging
from ht_chain.utils.string_util import regex_match, is_empty
import ast


def process_text(input_text: str) -> str:
    '''
    处理文本，去除HTML标签和转义字符
    :param input_text:
    :return:
    '''
    # 使用BeautifulSoup去除HTML标签
    plain_text = BeautifulSoup(input_text, "html.parser").get_text()
    # 还原转义字符
    plain_text = html.unescape(plain_text)
    return plain_text


@get_logger
def langchain_filter(logger, clargs):
    if clargs["content"] is None:
        raise ValueError("content is required")

    if clargs["filter_type"] == "rank":
        system_template = """You are a helpful assistant."""

        user_template = """/
        Given a question:
        [content]:
        ```
        {content}
        ```
        Rank the following 20 hints and determine whether this tip is helpful to solve the problem. 
        The higher the ranking, the more helpful the tips.
        [hints]:
        ```
        {hints}
        ```
        First explain why this sort, and then please output a sort of hints, in the front of the hints to solve the problem will help.
        The sorted hints are expressed in ```json``` format, where the key is the ranking and the value is the content of the hint.
        Unless the two hints are identical, each hint should appear in the final json.
        """
        # 形如：
        # ```json
        # {
        #     "0": [最相关的提示],
        #     "1": [第二相关的提示],
        #     ...
        # }
        # ```
        text = ({"content": clargs["content"], "hints": clargs["hints"]})
    elif clargs["filter_type"] == "score":
        system_template = """You are a helpful assistant."""
        # user_template = "{content} {signature} {hints}"
        user_template = """/
         Given a question:
        [content]:
        ```
        {content}
        ```
        Rate the following 20 hints, with only three choices: -1,0, and 1. 
        -1 indicates that the tip is not helpful in solving the problem, 1 indicates that the tip is helpful in solving the problem, and 0 indicates that the tip is not sure.
        [hints]:
        ```
        {hints}
        ```
        First explain why this score, and then output a score after the hints, with ```json``` format to express, json key is the content of the prompt, value is the score.
        Then sort the scores from 1 to 0 to -1.
        Unless the two hints are identical, each hint should appear in the final json.        
        """
        # 形如：
        # ```json
        # {
        #     "提示": "1",
        #     ...
        #         "提示": "0",
        # ...
        # "提示": "-1",
        # ...
        # }```
        text = ({"content": clargs["content"], "hints": clargs["hints"]})
    else:
        logger.error(f"filter_type is not supported: {clargs['filter_type']}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template),
    ])

    # text = ({"content": "a binary search code", "signature": "", "hints": ""})
    prompt = prompt_template.invoke(text)

    logger.info(prompt)

    if clargs["base_url"] == '' or clargs["base_url"] is None:
        model = ChatOpenAI(
            model=clargs["model"],
            api_key=clargs["api_key"],
            temperature=clargs["temperature"],
        )
    else:
        model = ChatOpenAI(
            model=clargs["model"],
            base_url=clargs["base_url"],
            api_key=clargs["api_key"],
            # temperature=clargs["temperature"],
        )
    parser = StrOutputParser()
    chain = model | parser

    while True:
        try:
            response = chain.invoke(prompt)
            logger.info(f"response: {response}")
            filter_hints = []

            json_str = find_code_snippet(response, "json")
            if json_str is None:
                continue

            filter_json = json.loads(json_str)
            if filter_json is None:
                continue

            if clargs["filter_type"] == "rank":
                for key in filter_json:
                    filter_hints.append(filter_json[key])
            elif clargs["filter_type"] == "score":
                for key in filter_json:
                    filter_hints.append(key)
            else:
                logger.error(f"filter_type is not supported: {clargs['filter_type']}")
                break

            if len(filter_hints) >= 10:
                break

            logger.info(f"fiter_hints: {filter_hints}")
        except Exception as e:
            logger.error(f"Error in langchain_filter: {e}")
            continue
    return filter_hints


@get_logger
def hint_filter(logger, clargs):
    """
    打分和过滤题目
    """
    index = 0

    if 'leetcode' in clargs['dir_path']:
        save_file_name = f"{clargs['left']}-{clargs['right']}_{clargs['filter_type']}"
        filter_full_info_path = f"{clargs['dir_path']}/{clargs['left']}-{clargs['right']}.json"
        save_res_path = f"{clargs['dir_path'].replace('full_info', 'hint_filter')}_{clargs['filter_type']}/{save_file_name}.json"

        logger.info(f"File with full_info Path: {os.path.abspath(clargs['full_info_path'])}")
        logger.info(f"Filtered File with full_info Path: {os.path.abspath(filter_full_info_path)}")
        logger.info(f"Generate results save to: {os.path.abspath(save_res_path)}")

        # filter_question_view = filter_questions(full_info_path, filter_full_info_path, difficulty, left, right)
        filter_question_view = filter_questions(clargs['full_info_path'], filter_full_info_path, [1, 2, 3],
                                                clargs['left'], clargs['right'])
        full_question_view_list = []

        for full_qv in tqdm(filter_question_view, desc="Filter the hint"):
            new_qv = full_qv.copy()
            # start_time_question = time.time()
            # try:
            index += 1
            len_filter_qv = len(filter_question_view)
            logger.info(color_message(f"Current question index is: {index}/{len_filter_qv} |"
                                      f" Leetcode formTitle: {full_qv['formTitle']} | Model: {clargs['model']}"))
            if not full_qv['codeSnippets']:
                logger.warning("Skip the locked question.")
                continue

            convers_args = {"content": process_text(full_qv['content']),
                            "hints": full_qv['infer_hints'],
                            "model": clargs['model'],
                            "base_url": clargs['url'],
                            "api_key": clargs['api_key'],
                            "filter_type": clargs['filter_type'],  # "rank" or "score"
                            }
            filtered_infer_hints = langchain_filter(convers_args)
            new_qv['infer_hints'] = filtered_infer_hints
            full_question_view_list.append(new_qv)
            # 获取原始提示字符串

        write_question_view_json_file(full_question_view_list, save_res_path)
    elif "apps" in clargs['dir_path']:  # data/apps/apps_end150_r20.json
        problems = jload(clargs['full_info_path'])
        # problems = problems[0:3]
        save_file_name = f"_{clargs['filter_type']}.json"
        total = len(problems)
        save_res_path = clargs['full_info_path'].replace('.json', save_file_name)
        new_problem_list = []

        for index, problem in enumerate(tqdm(problems, desc="Generate and test code:")):
            new_problem = problem.copy()
            index += 1
            problem_id = problem["problem_id"]
            difficulty = problem["difficulty"]
            question = problem["question"]
            new_problem["input_output"] = json.loads(problem["input_output"])
            new_problem["solutions"] = json.loads(problem["solutions"]) if problem["solutions"] else ""
            logger.info(color_message(
                f"Current question index is: {index}/{total} | APPS id: {problem_id} | difficulty: {difficulty}"))

            convers_args = {"content": process_text(question),
                            "hints": problem['infer_hints'],
                            "model": clargs['model'],
                            "base_url": clargs['url'],
                            "api_key": clargs['api_key'],
                            "filter_type": clargs['filter_type'],  # "rank" or "score"
                            }
            filtered_infer_hints = langchain_filter(convers_args)
            new_problem['infer_hints'] = filtered_infer_hints
            new_problem_list.append(new_problem)
            # 获取原始提示字符串

        write_question_view_json_file(new_problem_list, save_res_path)

    logger.info("--------------------------------------------")
    logger.info(f"Hint file: | {Path(args.full_info_path).absolute()}")
    logger.info(f"key: {args.url} | {args.api_key}")
    logger.info(f"Log file: | {Path(log_dir + log_file).absolute()}")
    logger.info(f"Filtered hint file: | {Path(save_res_path).absolute()}")
    return save_res_path


if __name__ == "__main__":
    # app()  # Automatically calls Typer to process command-line arguments
    parser = argparse.ArgumentParser(description="Process hint filtering")

    parser.add_argument('--dir_path', type=str, default='D:/software/workspace/python/langchain-hint/data/apps',
                        help="Data directory path")
    parser.add_argument('--full_info_path', type=str, default="apps_end150_r20.json",
                        help="Full info file path")
    parser.add_argument('--left', type=int, default=3000, help="Left boundary of question")
    parser.add_argument('--right', type=int, default=3001, help="Right boundary of question")
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help="Model (e.g., gpt-3.5-turbo-1106, gpt-4o-2024-08-06)")
    parser.add_argument('--filter_type', type=str, default='rank', help="rank or score")
    parser.add_argument("--url", default="https://api.key77qiqi.cn/v1", type=str, help="API base URL.")
    parser.add_argument("--api_key", default='sk-QXAmYyPDcNTRilmd192792Ee2331433aAd060151840fEc51', type=str,
                        help="API key for authentication.")

    args = parser.parse_args()

    if args.dir_path:
        args.full_info_path = f"{args.dir_path}/{args.full_info_path}"

    # Print arguments
    print("Received arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # Log setup
    date = datetime.now().strftime("%Y%m%d%H%M")
    log_dir = "log/hint_filter/"
    log_file = f"hint_filter_{date}.log"
    setup_logging(log_dir=log_dir, log_file=log_file)

    logger = logging.getLogger(__name__)

    # Call hint_filter with the parsed arguments
    syn_result_path = hint_filter(vars(args))

