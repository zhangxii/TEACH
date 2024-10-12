#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/10/8 9:51
# project: langchain-hint
# import bert_score
import logging

from datasets import tqdm
from openai import OpenAI

from ht_chain.utils.json_util import read_question_view_json_file, write_question_view_json_file
from ht_chain.utils.logger_util import get_logger, setup_logging
from ht_chain.utils.string_util import regex_match

client = OpenAI(
    api_key='sk-xx',  # qiqi key:zhangxi
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
def statistical_hint_helpful(logger, full_question_view_list: list):
    # full_question_view_list = full_question_view_list[0:2]
    if full_question_view_list is None:
        logger.warning("fullQuestionViewList is null")
        return
    ac_count = [0, 0, 0] # -1, 0, 1
    non_ac_count = [0, 0, 0] # -1, 0, 1
    for full_question_view in tqdm(full_question_view_list, desc="Score the hint helpful"):
        infer_hint_helpful_score = full_question_view['infer_hint_helpful_score']
        for i, generateResult in enumerate(full_question_view['generateResults']):
            logger.info(f"frontendQuestionId: {full_question_view['frontendQuestionId']} "
                        f"| index: {i+1} "
                        f"| status: {generateResult['runCodeCheckResult']['statusMsg']} "
                        f"| score: {infer_hint_helpful_score[i]['score']}")
            if generateResult['runCodeCheckResult']['statusMsg'] == "Accepted":
                ac_count[int(infer_hint_helpful_score[i]["score"]) + 1] += 1
            else:
                non_ac_count[int(infer_hint_helpful_score[i]["score"]) + 1] += 1

    logger.info(f"Accepted: {ac_count}")
    logger.info(f"Not Accepted: {non_ac_count}")


@get_logger
def statistical_hint_helpful_files(logger, hint_score_list: list, generate_list: list):
    if hint_score_list is None or generate_list is None:
        logger.warning("list is null")
        return
    if len(hint_score_list) != len(generate_list):
        logger.warning("The length of the two lists is not equal")
        return

    ac_count = [0, 0, 0]  # -1, 0, 1
    non_ac_count = [0, 0, 0]  # -1, 0, 1

    for (hint_score_fqv, generate_fqv) in tqdm(zip(hint_score_list, generate_list), desc="Score the hint helpful"):
        infer_hint_helpful_scores = hint_score_fqv['infer_hint_helpful_score']
        for i, generateResult in enumerate(generate_fqv['generateResults']):
            logger.info(
                f"frontendQuestionId: {generate_fqv['frontendQuestionId']} "
                f"| index: {i + 1} "
                f"| status: {generateResult['runCodeCheckResult']['statusMsg']} "
                f"| score: {infer_hint_helpful_scores[i]['score']}")
            if generateResult['runCodeCheckResult']['statusMsg'] == "Accepted":
                ac_count[int(infer_hint_helpful_scores[i]["score"]) + 1] += 1
            else:
                non_ac_count[int(infer_hint_helpful_scores[i]["score"]) + 1] += 1

    logger.info(f"Accepted: {ac_count}")
    logger.info(f"Not Accepted: {non_ac_count}")



@get_logger
def statistical_hint_code_consistency(logger, full_question_view_list: list):
    full_question_view_list = full_question_view_list[0:2]
    if full_question_view_list is None:
        logger.warning("fullQuestionViewList is null")
        return
    ac_count = [0, 0, 0]  # -1, 0, 1
    non_ac_count = [0, 0, 0]  # -1, 0, 1
    for full_question_view in tqdm(full_question_view_list, desc="Score the hint code consistency"):
        for i, generateResult in enumerate(full_question_view['generateResults']):
            infer_hint_code_consistency = int(generateResult['infer_hint_code_consistency'])
            logger.info(
                f"frontendQuestionId: {full_question_view['frontendQuestionId']} | index: {i + 1} "
                f"| status: {generateResult['runCodeCheckResult']['statusMsg']} | score: {infer_hint_code_consistency}")
            if generateResult['runCodeCheckResult']['statusMsg'] == "Accepted":
                ac_count[infer_hint_code_consistency + 1] += 1
            else:
                non_ac_count[infer_hint_code_consistency + 1] += 1

    logger.info(f"Accepted: {ac_count}")
    logger.info(f"Not Accepted: {non_ac_count}")


@get_logger
def statistical_hint_code_consistency_conditional(logger, hint_score_list: list,generate_list: list):
    if hint_score_list is None or generate_list is None:
        logger.warning("list is null")
        return
    if len(hint_score_list) != len(generate_list):
        logger.warning("The length of the two lists is not equal")
        return

    ac_count = [0, 0, 0]  # -1, 0, 1
    non_ac_count = [0, 0, 0]  # -1, 0, 1
    for (hint_score_fqv, generate_fqv) in tqdm(zip(hint_score_list, generate_list), desc="Score the hint code conditional consistency"):
        infer_hint_helpful_scores = hint_score_fqv['infer_hint_helpful_score']
        for i, generateResult in enumerate(generate_fqv['generateResults']):
            infer_hint_code_consistency = int(generateResult['infer_hint_code_consistency'])
            logger.info(
                f"frontendQuestionId: {generate_fqv['frontendQuestionId']} | index: {i + 1} "
                f"| status: {generateResult['runCodeCheckResult']['statusMsg']} | score: {infer_hint_code_consistency}")
            if infer_hint_helpful_scores[i]['score'] == "1":
                if generateResult['runCodeCheckResult']['statusMsg'] == "Accepted":
                    ac_count[infer_hint_code_consistency + 1] += 1
                else:
                    non_ac_count[infer_hint_code_consistency + 1] += 1

    logger.info(f"Accepted: {ac_count}")
    logger.info(f"Not Accepted: {non_ac_count}")


if __name__ == "__main__":
    setup_logging("../log/infer_hint_eval/")
    logger = logging.getLogger(__name__)

    generate_file_path = "../../data/leetcode/expr/res.json"

    full_question_view_list = read_question_view_json_file(generate_file_path)
    statistical_hint_helpful(full_question_view_list)
    statistical_hint_code_consistency(full_question_view_list)
