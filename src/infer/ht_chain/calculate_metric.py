#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/7/5 16:49
# project: langchain-hint

import logging
from typing import List

from ht_chain.utils.json_util import read_question_view_json_file
from ht_chain.utils.logger_util import get_logger, setup_logging


def count_accepted(statuses: List[str]) -> int:
    return statuses.count("Accepted")


@get_logger
def calculate_metric_show(logger, generate_file_path: str, lang: str, status_msg: str, k: int):
    pass_at_k = [0.0] * (k + 1)  # Initialize the array with 0
    full_question_view_list = read_question_view_json_file(generate_file_path)
    sample_size = len(full_question_view_list)
    full_size = sample_size

    for full_question_view in full_question_view_list:
        contain_lang = any(snippet["lang"].lower() == lang.lower() for snippet in full_question_view["codeSnippets"])
        # if not contain_lang:
        #     continue

        rcc_result_list = full_question_view["generateResults"]
        len_rcc_result_list = len(rcc_result_list)
        if len_rcc_result_list < k:
            logger.warning("FrontendQuestionId %s : n (%d) < k (%d), can not calculate pass@k. Passed!",
                                full_question_view["frontendQuestionId"], len_rcc_result_list, k)
            full_size -= 1
            continue

        status_msg_sum = sum(1 for rcc_result in rcc_result_list if
                             status_msg in rcc_result["runCodeCheckResult"]["statusMsg"])
        pass_rate = status_msg_sum / len_rcc_result_list

        for i in range(1, k + 1):
            pass_at_k[i] += estimator_pass_at_k(len_rcc_result_list, status_msg_sum, i)
        logger.debug(f" | passRate: {pass_rate} \t | pass@k {pass_at_k[k]} \t | {full_question_view['formTitle']}")

    for i in range(1, k + 1):
        pass_at_k[i] /= sample_size

    logger.info("Total size: %d | Calculated full size: %d", sample_size, full_size)
    show_pass_at_k(pass_at_k)


@get_logger
def show_pass_at_k(logger, pass_at_k: List[float]):
    k = len(pass_at_k) - 1
    sb = []
    sb.append("| pass@k |")
    for i in range(1, k + 1):
        sb.append(f"  k = {i}  |")
    sb.append("\n")

    sb.append("|--------|")
    for i in range(k):
        sb.append("---------|")
    sb.append("\n")

    sb.append("|        |")
    for i in range(1, k + 1):
        sb.append(f" {pass_at_k[i] * 100:.2f}%  |")
    sb.append("\n")

    logger.info("The pass@k value is as follows:\n%s", ''.join(sb))


def estimator_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0

    result = 1.0
    for i in range(n - c + 1, n + 1):
        result *= (1.0 - k / i)
    return 1.0 - result


@get_logger
def show_json_all_question_id(logger, generate_file_path: str):
    full_question_view_list = read_question_view_json_file(generate_file_path)
    question_ids = ' '.join(
        str(full_question_view["frontendQuestionId"]) for full_question_view in full_question_view_list)
    logger.info(f"Total size: {len(full_question_view_list)} | QuestionId: \n{question_ids}")


@get_logger
def show_csv_all_question_id(logger, csv_list: List[List[str]]):
    question_ids = ' '.join(csv_data[0] for csv_data in csv_list)
    logger.info(f"Total size: {len(csv_list)} | QuestionId: \n{question_ids}")


if __name__ == "__main__":
    setup_logging("../log/calculate_metric/")
    # 示例日志消息
    logger = logging.getLogger(__name__)

    qtype = 'easy'  # 题目难度
    difficulty = [1]  # 题目难度的筛选列表。list, 1是easy,2是medium,3是hard

    top_p = 1.0  # 生成代码的top_p
    temperature = 1.0  # 生成代码的temperature

    left = 3000  # 题目左边界
    right = 3140  # 题目右边界

    # origin, hint_ft_each_chain
    syn_type = 'hint_ft_each_chain'  # 生成代码的方式

    repeat = 10  # 每个题目生成和提交的重复实验次数
    time = 1  # 整个实验是第几次跑，这是个唯一且递增的id.如果repeat不一样，则time要重新开始计算
    k = repeat

    lang = 'java'  # 编程语言
    model = 'gpt-3.5-turbo-1106'  # gpt-3.5-turbo-1106,gpt-4-1106-preview 模型，gpt-3.5-turbo-1106, gpt-4-1106-preview
    dir_path = '../data/leetcode/full_info/hint_3000-3140_repeat20/'
    generate_file_path = f"{dir_path.replace('full_info', 'expr')}{qtype}_p{top_p:.1f}_t{temperature:.1f}_{left}-{right}_{model}_{syn_type}_r{repeat}_t{time}.json"

    # generate_file_path = "D:/BaiduSyncdisk/workspace/frangel-project/maven/FrAngel-template20230701-frame_constraint_genBlock(z3)/dataset/leetcode/expr/hint_3000-3140_repeat20/easy_p1.0_t1.0_3000-3140_gpt-4-1106-preview_origin_repeat10_time3.json"
    status_msg_filter = "Accepted"

    calculate_metric_show(generate_file_path, lang, status_msg_filter, k)
    # show_json_all_question_id(generate_file_path)