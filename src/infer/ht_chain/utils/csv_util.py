#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/7/8 22:19
# project: langchain-hint
import csv
import json

from typing import List

from ht_chain.utils.json_util import read_question_view_json_file
from ht_chain.utils.logger_util import get_logger


@get_logger
def write_data_to_csv(logger, generate_file_path: str, out_file_path: str, lang: str, repeat: int):
    full_question_view_list = read_question_view_json_file(generate_file_path)

    with open(out_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frontend Question ID', 'Title', 'Level', 'Generate Results'])

        for data in full_question_view_list:
            # Only consider the code snippet with the specified language
            contain_lang = any(snippet["lang"].lower() == lang.lower() for snippet in data['codeSnippets'])
            if not contain_lang:
                continue

            # Convert level to string representation
            level = {1: 'Easy', 2: 'Medium', 3: 'Hard'}.get(data['level'], 'Unknown')

            # Build the list of generate data
            syn_res = []
            accepted_num = 0
            for i in range(repeat):
                if data['generateResults'] and len(data['generateResults']) > i:
                    status_msg = data['generateResults'][i]["runCodeCheckResult"]["statusMsg"]
                    syn_res.append(status_msg)
                    if status_msg == 'Accepted':
                        accepted_num += 1
                else:
                    syn_res.append('')

            # Write data to CSV file
            writer.writerow([data['frontendQuestionId'], data['title'], str(level), ','.join(syn_res)])
    logger.info(f"CSV file saved to: {out_file_path}")


def read_csv_file(file_path: str) -> List[List[str]]:
    records = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            records.append(row)
    return records


