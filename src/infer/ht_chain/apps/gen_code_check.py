import io
import json
import random
import os
import pprint
import time
import multiprocessing
from datetime import datetime
from types import SimpleNamespace
from typing import Dict
import numpy as np

from ht_chain.apps.langchain_call_apps import get_langchain_completion
from ht_chain.utils.logger_util import get_logger, color_message, setup_logging
from ht_chain.utils.json_util import jload
from ht_chain.leetcode_submit_sh import find_code_snippet, format_code
from reindent import run as run_reindent
import testing_util as test_util
from ht_chain.apps.test_one_solution import check_correctness
from ht_chain.calculate_metric import estimator_pass_at_k, show_pass_at_k
from tqdm import tqdm
import argparse
import itertools

# for timing and debugging
TIMEOUT = 10
EXAMPLE_RESULTS = {"0": [[-2]], "1": [[False, False, False]], "2": [[True, True]], "3": [
    [False, True, False, True, False, False, False, True, False, True, False, True, True, True, False, True]],
                   "4": [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]}
EXAMPLE_ARGS = SimpleNamespace(debug=True)


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 10,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 10,
            "all-tabs": False
        }
    )

    return ret.getvalue()


def generate_prompt(args, test_case, prompt, solutions, tokenizer, starter_code=None):
    """
    :param args:
    :param test_case: 传入只是为了获取是否有starter_code
    :param prompt:
    :param solutions:
    :param tokenizer:
    :param starter_code:
    :return:
    """
    _input = "\nQUESTION:\n"
    data = prompt
    _input += data
    if starter_code is not None:
        data = starter_code
        data = "\n" + data
        _input += data

    # data = test_case  #hint:origin method
    data = test_case

    _input += """ \n If  is Standard Input Format, use input() for input and print() for output, do not generate main. 
                If the Format is Call-Based, use starter_code as the function name and generate the function only. \n"""
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"
    else:
        _input += "\nUse Call-Based format"

    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        sample_sol = random.choice(solutions)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol


@get_logger
def print_results(logger, results: Dict, args: argparse.Namespace = None):
    """
    Given the results evaluated against the testcases we output some statistics.

    >>> print_results(EXAMPLE_RESULTS, EXAMPLE_ARGS)
    number of compile errors = 1 avg = 0.2
    number of runtime errors = 1 avg = 0.2
    number of test cases run = 5
    Test Case Average (average accuracy over problems) = 0.3
    Strict Accuracy (all test cases passed / total problems) = 0.2
    """
    res = []
    per_prob_res = []
    all_correct = []

    k = args.repeat
    pass_at_k = [0.0] * (k + 1)

    for index in results:
        correct = 0
        for single_res in results[index]:
            np_single = np.asarray([single_res])
            res.extend(np_single)
            per_prob_res.append(np.mean(np_single > 0))
            all_correct.append(np.all(np_single > 0))
            correct += np.all(np_single > 0)
        logger.debug(f"Problem {index} | Correct: {correct} | Total: {len(results[index])}")

        for i in range(1, k + 1):
            pass_at_k[i] += estimator_pass_at_k(args.repeat, correct, i)

    # We count both compile errors and runtime errors for multiple tests as one error.
    compile_errors = len([e for e in res if -2 in e])
    runtime_errors = len([e for e in res if -1 in e])
    total_testcases = len(res)
    if args and args.debug:
        print(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases}")
        print(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}")
        print(f"number of test cases run = {total_testcases}")

    print(f"Test Case Average (average accuracy over problems) = {np.mean(per_prob_res)}")
    print(f"Strict Accuracy (all test cases passed / total problems) = {np.mean(all_correct)}")
    for i in range(1, k + 1):
        pass_at_k[i] /= len(results)
    show_pass_at_k(pass_at_k)


@get_logger
def main(logger, args):
    argsdict = vars(args)
    logger.info(pprint.pformat(argsdict))
    problem_list = []
    results = {}

    problems = jload(args.input)
    # problems = problems[0:3]

    total = len(problems)
    lang = "python"
    save_name = f"{args.model}_{args.syn_type}_r{args.repeat}_t{args.time}"
    problem_list_loc = f"{args.save}{save_name}.json"
    results_loc = f"{args.save}{save_name}_res.json"

    if args.reuse:
        if os.path.exists(problem_list_loc):
            with open(problem_list_loc, "r") as f:
                problem_list = json.load(f)
        if os.path.exists(results_loc):
            with open(results_loc, "r") as f:
                results = json.load(f)
    else:
        for index, problem in enumerate(tqdm(problems, desc="Generate and test code:")):
            problem_id = problem["problem_id"]
            difficulty = problem["difficulty"]
            test_case = json.loads(problem["input_output"])
            prompt = problem["question"]
            starter_code = problem["starter_code"]
            problem["solutions"] = json.loads(problem["solutions"]) if problem["solutions"] else ""
            problem["input_output"] = json.loads(problem["input_output"])

            if not starter_code:
                starter_code = None

            prompt_text, sample_sol = generate_prompt(args, test_case, prompt, problem["solutions"], None, starter_code)
            logger.debug("PROMPT_TEXT:")
            logger.debug(prompt_text)

            logger.info(
                color_message(f"difficulty: {difficulty} | problem_id: {problem_id} | index: {index + 1}/{total}"))
            problem_results = []
            curr_res = [-2]
            for i in range(1, args.repeat + 1):
                problem['generateResults'] = []
                logger.info(
                    f"difficulty: {difficulty} | problem_id: {problem_id}| index: {index + 1}/{total} | repeat: {i}/{args.repeat}")
                hints = {
                    'origin': 'origin',
                    'template': 'template',
                    'hint_ft_each': problem['infer_hints'][i - 1],
                    'hint_ft_each_chain': problem['infer_hints'][i - 1],
                }

                start = time.time()
                convers_args = {"content": prompt_text,
                                "syn_type": args.syn_type,
                                "language": lang,
                                "signature": starter_code,
                                "hints": hints[args.syn_type],
                                "model": args.model,
                                "base_url": args.url,
                                "api_key": args.api_key,
                                "temperature": args.temperature
                                }
                output_str = ""
                # output_str = problem["solutions"][0]
                while True:
                    try:
                        response = get_langchain_completion(convers_args)
                        output_str = response
                    except Exception as e:
                        logger.warning("Unexpected exception in generating solution")
                        logger.warning(e)
                        output_str = "code"
                    end = time.time()

                    if args.peeking == 1.0:
                        output_str = sample_sol
                        break
                    elif len(output_str):
                        code = find_code_snippet(output_str, lang)
                        output_str = format_code(code, lang)
                        if output_str:
                            break
                logger.info(f"index:{index + 1} |formated code:\n{output_str}")
                # gpt_codes[index + args.start] = output_str
                # hint: 检查结果
                # res = check_correctness(problem, generation=output_str, timeout=TIMEOUT, debug=args.debug)

                try:
                    curr_res = check_correctness(problem, generation=output_str, timeout=TIMEOUT, debug=args.debug)
                    fixed = []
                    for e in curr_res:
                        if isinstance(e, np.ndarray):
                            e = e.item(0)
                        if isinstance(e, np.bool_):
                            e = bool(e)
                        fixed.append(e)
                    curr_res = fixed
                    if not np.all(curr_res):
                        print(f"Results were not all True: {curr_res}")
                except Exception as e:
                    print(f"test framework exception = {repr(e)}{e}\n")
                    break
                finally:
                    assert isinstance(curr_res, list)
                    problem_results.append(curr_res)
                # res = "false"
                # logger.info(f"print {problem_id}: {res} | i:{i}| model:{args.model} |syn_type:{args.syn_type}| generateCode:{output_str}")
                results[problem_id] = problem_results  # key是问题的id #todo:这里有bug
                problem['generateResults'].append({'index': i,
                                                   'generateModel': args.model,
                                                   'synType': args.syn_type,
                                                   'hint': hints[args.syn_type],
                                                   'generateCode': output_str,
                                                   'runCodeCheckResult': curr_res})
                # logger.info(color_message(f"status_msg: {res}"))
            if args.debug:
                logger.debug(f"Generation time: {end - start}")
                logger.debug(f"Generated output string:")
                logger.debug(output_str)
                logger.debug("------------------------------------------------------------")
            problem_list.append(problem)

        logger.info(f"Saving generate and check res to: {problem_list_loc} ")
        with open(problem_list_loc, "w") as f:
            json.dump(problem_list, f, indent=2)

        logger.info(f"Saving results to: {results_loc} ")
        with open(results_loc, "w") as f:
            json.dump(results, f, indent=2)

    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    res_markdown = f"""Save expriment result:\n 
(){save_name}\n
| name  | {problem_list_loc.split('.json')[0]}    |
| ----- | ------- |
| input  |{args.input}    |
| path  |{os.path.abspath(problem_list_loc)}    |
| time  |{cur_time}      |
| size  |{len(problem_list)}               |
| key   |url:{args.url}, key: {args.api_key}  |
| model |model={args.model}, temperature={args.temperature}  |\n"""
    logger.info(res_markdown)

    # for test
    # results = dict(itertools.islice(results.items(), 3))

    # 假设 results 是一个 map (dictionary)
    logger.info(f"Show interview the results:")
    sub_results = {k: v for k, v in results.items() if 0 <= int(k) < 3000}

    logger.info(f"Show introductory the results:")
    sub_results = {k: v for k, v in results.items() if 4000 <= int(k) < 5000}
    print_results(sub_results, args)

    logger.info(f"Show competition the results:")
    sub_results = {k: v for k, v in results.items() if 3000 <= int(k) < 4000}
    print_results(sub_results, args)

    logger.info(f"Show all the results:")
    print_results(results, args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and test Python code using a trained model.")
    parser.add_argument("--input", default="../data/apps/apps_end150_r20_rank.json", type=str,
                        help="The input file of apps subset.")
    parser.add_argument("--save", type=str, default="./data/apps/apps_end150_r20_res/")
    parser.add_argument("--reuse", default=False, type=bool, help="reuse the result.")
    parser.add_argument("--model", default="gpt-3.5-turbo-1106", type=str, help="Model name.")
    parser.add_argument("--url", default="https://api.key77qiqi.cn/v1", type=str, help="API base URL.")
    parser.add_argument("--api_key", default='sk-QXAmYyPDcNTRilmd192792Ee2331433aAd060151840fEc51', type=str,
                        help="API key for authentication.")
    parser.add_argument("--temperature", default=1, type=float, help="Sampling temperature.")
    parser.add_argument("--syn_type", default="origin", type=str, help="Synthesis type.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--time", type=int, default=1)
    parser.add_argument("-t", "--test_loc", default="~/apps/data_split/test.json", type=str,
                        help="Path to the test folder.")
    parser.add_argument("-r", "--root", default="../", type=str, help="Where the data is stored.")
    parser.add_argument("-l", "--load", default="", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument("-e", "--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--split", type=str, default="test", help="What split to use.")

    date = datetime.now().strftime("%Y%m%d")
    log_dir = "../log/apps_gen_code/"
    setup_logging(log_dir=log_dir)

    args = parser.parse_args()

    main(args)

