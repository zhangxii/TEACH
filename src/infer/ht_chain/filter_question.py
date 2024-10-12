#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/7/5 16:49
# project: langchain-hint
import json
import logging
from typing import List, Dict, Any

from ht_chain.utils.json_util import read_question_view_json_file, write_question_view_json_file

logger = logging.getLogger("FilterQuestions")


# 读取和写入JSON文件的函数
# def read_full_question_view_json_file(file_path: str) -> List[FullQuestionView]:
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#         return [FullQuestionView(**item) for item in data]
#
#
# def write_full_question_view_json_file(data: List[FullQuestionView], file_path: str):
#     with open(file_path, 'w', encoding='utf-8') as file:
#         json.dump([item.__dict__ for item in data], file, indent=4, ensure_ascii=False)
#         logger.info(f"JSON written to: {file_path}")




# 检查问题ID是否在指定范围内
def is_id_in_range(question, id_range: List[int]) -> bool:
    if question["frontendQuestionId"] is not None:
        question_id = int(question["frontendQuestionId"])
        return id_range[0] <= question_id <= id_range[1]
    return False

def contain_lang(question, lang: str) -> bool:
    if question["codeSnippets"] is None:
        # logger.info("Skip the locked question.")
        return False
    return any(snippet['lang'].lower() == lang.lower() for snippet in question["codeSnippets"])


def filter_questions_full_info(full_info_path: str, filter_full_info_path: str,
                               difficulty: List[int], id_range: List[int]) -> list[dict[str, Any]]:
    lang = "java"
    full_question_view_list = read_question_view_json_file(full_info_path)
    filter_list = [q for q in full_question_view_list if q["level"] in difficulty and
                   is_id_in_range(q, id_range) and contain_lang(q, lang)]
    write_question_view_json_file(filter_list, filter_full_info_path)
    logger.info("Filtered FullInfo File size: %d", len(filter_list))
    return filter_list


def filter_questions(full_info_path: str, filter_full_info_path: str, difficulty: List[int], left: int, right: int) -> \
        list[dict[str, Any]]:
    id_range = [left, right]
    return filter_questions_full_info(full_info_path, filter_full_info_path, difficulty, id_range)
