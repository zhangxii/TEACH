#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/7/6 11:07
# project: langchain-hint
import io
import json
import logging
import os


from typing import List, Dict, Any

from tqdm import tqdm

from ht_chain.utils.logger_util import get_logger

logger = logging.getLogger("JsonUtil")


@get_logger
def read_question_view_json_file(logger, file_path: str) -> List[Dict[str, Any]]:
    logger.info(f"Read json from: \n{os.path.abspath(file_path)}")
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


@get_logger
def write_question_view_json_file(logger, data: List[Dict[str, Any]], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        logger.info(f"Write json to: \n{os.path.abspath(file_path)}")



def _make_w_io_base(f, mode: str, encoding: str = 'utf-8'):
    # 用于创建文件对象写文件，确保文件路径存在，并以正确的模式打开文件
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding=encoding)
    return f


def _make_r_io_base(f, mode: str, encoding: str = 'utf-8'):
    # 用于创建文件对象写文件，确保文件路径存在，并以正确的模式打开文件
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding=encoding)
    return f


def jdump(obj, f, mode="w", indent=4, default=str, ensure_ascii = True, encoding = "utf-8"):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
        ensure_ascii: Whether to encode strings as ASCII characters # hint add
        encoding: The encoding type of file; defaults to `utf-8`# hint add
    """
    f = _make_w_io_base(f, mode, encoding=encoding)
    if isinstance(obj, (dict, list)):
        # json.dump(obj, f, indent=indent, default=default)
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=ensure_ascii)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r", encoding="utf-8"):
    """
    Load a .json file into a dictionary.
    hint: add encoding type.
    @param f:
    @param mode:
    @param encoding:
    @return: a json dictionary
    """
    f = _make_r_io_base(f, mode, encoding=encoding)
    jdict = json.load(f)
    f.close()
    return jdict


def select_n_data(input_file, output_file, n_data):
    """
    从输入文件中选择前n_data个数据，存到输出文件中
    @param input_file:
    @param output_file:
    @param n_data:
    @return:
    :author: zhangyating
    """
    json = jload(input_file)
    sub_json = []
    for i, sample in tqdm(enumerate(iter(json))):
        if i >= n_data:
            break
        sub_json.append(sample)

    jdump(sub_json, output_file, ensure_ascii=False, encoding="utf-8")
    print(f'data saved to {output_file}')


def jsonl_2_json(jsonl_file, json_file):
    """
    将jsonl文件转换为json list
    @param jsonl_file:
    @param json_file:
    @return:
    """

    # 读取文件内容
    with open(jsonl_file, 'r', encoding="utf-8") as infile:
        jsonl_data = infile.readlines()

    # 将 jsonl 文件内容转换为 JSON 格式
    json_data = []
    for line in jsonl_data:
        json_data.append(json.loads(line))

    jdump(json_data, json_file, ensure_ascii=False, encoding="utf-8")
    print(f'data saved to {json_file}')


def jsonline_dump(jsonl_data, output_file, ensure_ascii=False, encoding="utf-8"):
    """
    将 JSON Lines 数据写入文件中。
    :jsonl_data (list): 包含 JSON 对象的列表。
    :output_file (str): 输出文件的路径。
    :ensure_ascii (bool, optional): 是否将非 ASCII 字符转义为 ASCII。默认为 False。
    :encoding (str, optional): 输出文件的编码。默认为 "utf-8"。
    """
    with open(output_file, 'a', encoding=encoding) as outfile:
        for line in jsonl_data:
            outfile.write(json.dumps(line,ensure_ascii=ensure_ascii) + "\n")

    outfile.close()


def jsonline_load(json_file, encoding="utf-8"):
    """
    从 JSON Lines 文件中加载数据并返回一个包含 JSON 对象的列表。
    :json_file (str): JSON Lines 文件的路径。
    :encoding (str, optional): 文件编码。默认为 "utf-8"。
    Returns:
        list: 包含从文件中加载的 JSON 对象的列表。
    """
    jsonl_data = []
    with open(json_file, 'r', encoding=encoding) as infile:
        str_data = infile.readlines()
    for element in str_data:
        jsonl_data.append(json.loads(element))
    infile.close()
    return jsonl_data


def read_txt_file(path):
    f = _make_w_io_base(path, "r", encoding="utf-8")
    content = f.read()
    f.close()
    return content



