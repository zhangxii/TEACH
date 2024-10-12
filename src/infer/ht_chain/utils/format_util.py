#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/7/8 10:38
# project: langchain-hint
import black
import re

def format_code(unformatted_code: str) -> str:
    # 还原转义字符
    unformatted_code = (
        unformatted_code.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\\"", "\"")
        .replace("\\'", "'")
        .replace("\\\\", "\\")
    )
    # 使用 black 格式化代码
    formatted_code = black.format_str(unformatted_code, mode=black.FileMode())
    return formatted_code