#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/7/8 10:40
# project: langchain-hint
import re


def regex_match(input_text: str, pattern: str) -> str:
    # 编译正则表达式
    compiled_pattern = re.compile(pattern)
    # 查找匹配项
    match = compiled_pattern.search(input_text)
    # 返回匹配的内容
    if match:
        return match.group(0)
    else:
        return ""


def is_empty(input_text: str) -> bool:
    return not input_text or len(input_text) == 0