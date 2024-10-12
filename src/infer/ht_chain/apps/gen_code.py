import io
import json
import random
import os
import pprint
import time

from datasets import load_dataset
from datetime import datetime

from langchain_call_apps import get_langchain_completion
from ht_chain.utils.logger_util import get_logger, color_message, setup_logging
from ht_chain.leetcode_submit_sh import find_code_snippet, format_code
from reindent import run as run_reindent

# 新的API模型调用相关库
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import ast

# for timing and debugging
from datetime import datetime, date
from tqdm import tqdm


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
    _input = "\nQUESTION:\n"
    data = prompt
    _input += data
    if starter_code != None:
        data = starter_code
        data = "\n" + data  # + "\n"
        _input += data
    else:
        # _input += "\n\n"
        pass

    data = test_case
    
    _input +=""" \n If  is Standard Input Format, use input() for input and print() for output, do not generate main. 
                If the Format is Call-Based, use starter_code as the function name and generate the function only. \n"""
    if not data.get("fn_name"):
        _input += "\nUse Standard Input format"  # \n"
    else:
        _input += "\nUse Call-Based format"  # \n"

    _input += "\nANSWER:\n"

    if args.peeking > 0.0:
        # 当 peeking 设置为大于 0.0 的值时，模型会看到部分已知的解决方案。具体来说，代码会根据 peeking 参数的值，
        # 截取某个已知解决方案的一部分，将其作为提示提供给模型。
        # 例如，如果 peeking = 0.3，则模型可能会看到一个解决方案的前 30% 的代码。这些代码会作为生成提示的一部分，帮助模型生成后续代码。
        # Need to do some peeking.

        # Read one example solution
        sols = solutions

        # Choose the shortest solution for the model to use.
        # This is so we can conserve tokens (1024 max)
        # sample_sol = min(sols, key=len)

        # # Add args.peeking% of that solution to the prompt
        # sample_sol_token_ids = tokenizer.encode(sample_sol, verbose=False)
        # num_to_keep = int(len(sample_sol_token_ids) * args.peeking)
        # sample_sol_token_ids = sample_sol_token_ids[:num_to_keep]
        # _input += tokenizer.decode(sample_sol_token_ids)

        # Alternatively take a random solution
        sample_sol = random.choice(sols)
        rand_sol = reindent_code(sample_sol)
        rand_sol = tokenizer.encode(rand_sol, verbose=False)
        tokens_taken = int(args.peek_frac * len(rand_sol))
        rand_sol = rand_sol[:tokens_taken]
        _input += tokenizer.decode(rand_sol)
    else:
        sample_sol = None

    return _input, sample_sol

@get_logger
def main(logger, args):
    argsdict = vars(args)
    logger.info(pprint.pformat(argsdict))

    problems = load_dataset("codeparrot/apps", split=f"{args.split}", trust_remote_code=True)

    gpt_codes = {}

    if not os.path.exists(args.save):
        os.makedirs(args.save, exist_ok=True)
    if not args.end:
        codes_loc = os.path.join(args.save, f"all_codes_{args.syn_type}_r{args.repeat}.json")
    else:
        codes_loc = os.path.join(args.save, f"{args.start}-{args.end}_codes_{args.syn_type}_r{args.repeat}.json")

    if args.index:
        problems = load_dataset("codeparrot/apps", split=f"{args.split}[{args.index}]", trust_remote_code=True)
    else:
        if args.start > len(problems) or args.start < 0:
            logger.info(f"start index {args.start} > number of problems {len(problems)}")
            return
        start = args.start
        if args.end is None or args.end > len(problems):
            end = len(problems)
        else:
            end = args.end
        problems = load_dataset("codeparrot/apps", split=f"{args.split}[{start}:{end}]", trust_remote_code=True)

    # 使用新的API模型
    # model = ChatOpenAI(
    #     model=args.model,
    #     base_url=args.base_url,
    #     api_key=args.api_key,
    #     temperature=args.temperature,
    # )
    # parser = StrOutputParser()
    # chain = model | parser
    lang = "python"
    # main eval loop
    for index, problem in enumerate(tqdm(problems)):
        problem["solutions"] = json.loads(problem["solutions"])
        problem["input_output"] = json.loads(problem["input_output"])
        test_case = problem["input_output"]
        prompt = problem["question"]
        starter_code = problem["starter_code"]
        solutions = problem["solutions"]
        if not starter_code:
            starter_code = None

        # Read the question in
        prompt_text, sample_sol = generate_prompt(args, test_case, prompt, solutions, None, starter_code)
        logger.debug("PROMPT_TEXT:")
        logger.debug(prompt_text)
        
        hints = {
                'origin': 'origin',
                'template': 'template',
                # 'hint_ft_each': full_qv['infer_hints'][i - 1],
                # 'hint_ft_each_chain': full_qv['infer_hints'][i - 1],
            }
        
        # 通过新的API调用生成代码
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
        while True:
            try:
                # response = chain.invoke(prompt_text)
                response = get_langchain_completion(convers_args)
                output_str = response
            except Exception as e:
                logger.warning("Unexpected exception in generating solution")
                logger.warning(e)
                # Default to empty string on errors
                output_str = ""
            end = time.time()



            if args.peeking == 1.0:
                output_str = sample_sol
                break
            elif len(output_str):
                # output_str = output_str.split("ANSWER:\n")[1].replace("<|endoftext|>", "")
                code = find_code_snippet(output_str, lang)
                output_str = format_code(code, lang)
                if output_str:
                    logger.info(color_message(f"index:{index} |formated code:{code}"))
                    break

        # Save the generated sol
        gpt_codes[index + args.start] = output_str

        if args.debug:
            logger.debug(f"Generation time: {end - start}")
            logger.debug(f"Generated output string:")
            logger.debug(output_str)
            logger.debug("------------------------------------------------------------")
    logger.info(f"Saving to: {codes_loc} ")

    with open(codes_loc, "w") as f:
        json.dump(gpt_codes, f,indent = 2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a tranined model to generate Python code.")
    parser.add_argument("--model", default="gpt-4o", type=str, help="Model name.")
    parser.add_argument("--url", default="", type=str, help="API base URL.")
    parser.add_argument("--api_key", default='sk-xx', type=str,
                        help="API key for authentication.")
    parser.add_argument("--temperature", default=1, type=float, help="Sampling temperature.")
    parser.add_argument("--syn_type", default="origin", type=str, help="Synthesis type.")
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("-t", "--test_loc", default="~/apps/data_split/test.json", type=str,
                        help="path to the test folder.")
    parser.add_argument("-r", "--root", default="../", type=str, help="where the data is stored.")
    parser.add_argument("-l", "--load", default="", type=str)
    parser.add_argument("--peeking", default=0.0, type=float)
    parser.add_argument("--num-beams", default=5, type=int)
    parser.add_argument("-s", "--start", default=0, type=int)
    parser.add_argument("-e", "--end", default=None, type=int)
    parser.add_argument("-i", "--index", default=None, type=int)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("--split", type=str, default="test", help="What split to use.")
    parser.add_argument("--save", type=str, default="./data")

    date = datetime.now().strftime("%Y%m%d")
    log_dir = "../log/apps_gen_code/"
    setup_logging(log_dir=log_dir)
    
    args = parser.parse_args()

    main(args)
