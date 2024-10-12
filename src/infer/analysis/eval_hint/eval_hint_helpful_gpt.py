# import bert_score
import logging

from datasets import tqdm
from openai import OpenAI

from analysis.eval_hint.statistical_results import statistical_hint_helpful
from ht_chain.utils.json_util import read_question_view_json_file, write_question_view_json_file
from ht_chain.utils.logger_util import get_logger, setup_logging
from ht_chain.utils.string_util import regex_match

client = OpenAI(
    api_key='xx',
)

@get_logger
def get_completion(logger, prompt, model="gpt-4o", temperature=1, topp=1):
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]
    # openai api v 1.31
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=topp
    )
    return response.choices[0].message.content

@get_logger
def get_score(logger,infer_hints, response):
    """
    查找响应里的代码片段
    :param response:
    :param lang:
    :return:
    """
    regex = f"(?<=```)([\s\S]*?)(?=```)"
    score = regex_match(response, regex)
    hint_helpful_score = score.split(",")
    score_list = []

    for i, (hint, score) in enumerate(zip(infer_hints, hint_helpful_score)):
        score_dict = {}
        score_dict["index"] = i+1
        score_dict["hint"] = hint
        score_dict["score"] = score
        score_list.append(score_dict)
    return score_list


@get_logger
def eval_hint_helpful(logger, generate_file_path: str, save_path: str):
    full_question_view_list = read_question_view_json_file(generate_file_path)
    full_question_view_list = full_question_view_list[0:2]
    if full_question_view_list is None:
        logger.warning("fullQuestionViewList is null")
        return
    fqv_hint_helpful_score_list = []
    for full_question_view in tqdm(full_question_view_list, desc="Score the hint helpful"):
        copy_fq = full_question_view.copy()
        # 一对多
        # 那个数量多的要再套一层列表
        question = full_question_view['content']
        infer_hints = full_question_view['infer_hints'][0:10]
        # if int(full_question_view['frontendQuestionId']) == id:
        if infer_hints and len(infer_hints) > 0:
            prompt = (f"Given a programming task as follows: \n "
                      f"----------------------------------------------\n"
                      f"{question} \n "
                      f"Rate the following ten tips on a scale of -1, 0, or 1. "
                      f"-1 means that the hint is not helpful in solving the problem, "
                      f"1 means that the hint is helpful in solving the problem, "
                      f"and 0 means that it is not sure whether it is helpful. \n"
                      f"----------------------------------------------\n"
                      f"{infer_hints}"
                      f"----------------------------------------------\n"
                      f"No explanation needed, just give me ten scores in triple quotes separated by commas. "
                      f"For example: ```0,0,1,0,1,-1,-1,-1,0,0```")
            response = get_completion(prompt)

            copy_fq['infer_hint_helpful_score'] = get_score(infer_hints, response)
        fqv_hint_helpful_score_list.append(copy_fq)
        logger.info(f"Score: \n {copy_fq['infer_hint_helpful_score']}")

    statistical_hint_helpful(fqv_hint_helpful_score_list)
    if save_path:
        write_question_view_json_file(fqv_hint_helpful_score_list, save_path)
        logger.info("Saving to file: " + save_path)



if __name__ == "__main__":
    setup_logging("../log/infer_hint_eval/")
    logger = logging.getLogger(__name__)
    generate_file_path = "../../data/leetcode/expr/res.json"
    save_path = generate_file_path.replace(".json", "_hint_helpful_score.json")
    eval_hint_helpful(generate_file_path, save_path)