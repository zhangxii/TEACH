# import bert_score
import logging

from datasets import tqdm
from openai import OpenAI

from ht_chain.utils.json_util import read_question_view_json_file, write_question_view_json_file
from ht_chain.utils.logger_util import get_logger, setup_logging
from ht_chain.utils.string_util import regex_match

client = OpenAI(
    api_key='sk-xx',
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
def get_score(logger, response):
    """
    查找响应里的代码片段
    :param response:
    :param lang:
    :return:
    """
    regex = f"(?<=```)([\s\S]*?)(?=```)"
    score = regex_match(response, regex)
    return score



@get_logger
def eval_hint_code_consistency(logger, generate_file_path: str, save_path: str):
    full_question_view_list = read_question_view_json_file(generate_file_path)
    full_question_view_list = full_question_view_list[0:1]
    if full_question_view_list is None:
        logger.warning("fullQuestionViewList is null")
        return
    save_infer_hint_code_consistency_list = []
    for full_question_view in tqdm(full_question_view_list, desc="Score the hint helpful"):

        copy_fq = full_question_view.copy()
        copy_fq['generateResults'] = []
        # 一对多
        # 那个数量多的要再套一层列表
        question = full_question_view['content']
        infer_hints = full_question_view['infer_hints']
        for i, generateResult in enumerate(full_question_view['generateResults']):
            copy_generateResult = generateResult.copy()
            generateCode = generateResult['generateCode']
            infer_hint = infer_hints[i]
            prompt = (f"Given a question, a hint, and a code:\n "
                      f"-------------------question---------------------------\n"
                      f"{question} \n "
                      f"-------------------question---------------------------\n"
                      f"-------------------hint---------------------------\n"
                      f"{infer_hint} \n "
                      f"-------------------hint---------------------------\n"
                      f"-------------------code---------------------------\n"
                      f"{generateCode} \n "
                      f"-------------------code---------------------------\n"
                      f"Rate on a scale of -1, 0, or 1. "
                      f"1 means if you think the code follows the hint to solve the problem,, "
                      f"- 1 means if you think the code doesn't follow the hint to solve the problem, "
                      f"and 0 means that it is not sure whether it is follow. \n"
                      f"No explanation needed, just give me scores in triple quotes separated by commas. "
                      f"For example: ```1```")
            response = get_completion(prompt)
            copy_generateResult['infer_hint_code_consistency'] = get_score(response)
            logger.info(f"Score: \n {copy_generateResult['infer_hint_code_consistency']}")
            copy_fq['generateResults'].append(copy_generateResult)
    save_infer_hint_code_consistency_list.append(copy_fq)


    if save_path:
        write_question_view_json_file(save_infer_hint_code_consistency_list, save_path)
        logger.info("Saving to file: " + save_path)

if __name__ == "__main__":
    setup_logging("../log/infer_hint_eval/")
    logger = logging.getLogger(__name__)
    generate_file_path = "../../data/leetcode/expr/res.json"
    save_path = generate_file_path.replace(".json", "_hint_code_consistency.json")
    eval_hint_code_consistency(generate_file_path, save_path)