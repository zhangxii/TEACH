#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/6/20 11:45
# project: langchain-hint

import argparse
import logging
import os

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from ht_chain.utils.logger_util import get_logger, setup_logging



def readFile(prompt_file_path):
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        return file.read()


@get_logger
def get_langchain_completion(logger, clargs):
    if clargs["content"] is None:
        raise ValueError("content is required")
    logger.info(f"syn_type: {clargs['syn_type']}")
    if clargs['syn_type'] == 'origin': # if hints is None, it is a origin code generation task
        system_template = """/
            You are a helpful assistant to generate code. 
        """
        user_template ="""/
                [content]:
                ```
                {content}
                ```
                """
        text = ({"content": clargs["content"], "signature": clargs["signature"], 'language': clargs['language']})
    elif clargs['syn_type'] == 'hint_ft_each' or clargs['syn_type'] == 'hint_ori':
        system_template = "You are a helpful assistant to generate code."
        user_template = """/
                [content]:
                ```
                {content}
                ```
                [hints]:
                ```
                {hints}
                ```
                Pleasce using hints to help you think about the solution.
        """
        text = ({"content": clargs["content"], "signature": clargs["signature"], 'language': clargs['language'], "hints": clargs["hints"]})
    elif clargs['syn_type'] == 'template':
        system_template = """
            You are a helpful assistant to generate code.
            """
        # user_template = "{content} {signature} {hints}"
        user_template = """/
            Human programmers typically follow several steps to solve programming competition problems:
            1.   **Read the problem**: Carefully read the problem description to understand the requirements, input/output format, and constraints.   
                Make sure to grasp the meaning of the problem and not overlook any details.   Abstract the key concepts.
            2.   **Analyze the problem**: Analyze the essence of the problem based on its description.  
                Determine the problem category (such as greedy algorithms, dynamic programming, graph theory, mathematics, etc.) and think about the solution approach.
            3.   **Design the algorithm**: Design a suitable algorithm to solve the problem based on the analysis.   This step may include the following sub-steps:
            - **Conceptualize the solution**: Based on the problem type and past experience, brainstorm a solution and verify its feasibility.
            - **Choose data structures**: Select appropriate data structures like arrays, linked lists, stacks, queues, hash tables, etc., based on the algorithm's requirements.
            - **Estimate complexity**: Estimate the algorithm's time and space complexity to ensure it can run within a reasonable time under the given constraints.
            4.   **Write code**: Implement the designed algorithm in code.   When writing code, focus on readability and adherence to coding conventions to minimize errors.
            
            The task information as follow:
            [content]:
            ```
            {content}
            ```
            For this above question, write a solution with ```{language}``` tag. Please generate code by referencing the ways:
            1.the steps of programmers would solve a problem on a programming competition.
            """
            # please generate code by referencing the chain of how human programmers would approach a problem on a programming competition.
        text = ({"content": clargs["content"], "signature": clargs["signature"], 'language': clargs['language']})
    elif clargs['syn_type'] == 'hint_ft_each_chain' or clargs['syn_type'] == 'hint_ori_chain':
        system_template = """
            You are a helpful assistant to generate code.   
            """        
        # user_template = "{content} {signature} {hints}"
        user_template = """/   
            Human programmers typically follow several steps to solve programming competition problems:
            1.   **Read the problem**: Carefully read the problem description to understand the requirements, input/output format, and constraints.   
                Make sure to grasp the meaning of the problem and not overlook any details.   Abstract the key concepts.
            2.   **Analyze the problem**: Analyze the essence of the problem based on its description.  
                Determine the problem category (such as greedy algorithms, dynamic programming, graph theory, mathematics, etc.) and think about the solution approach.
            3.   **Design the algorithm**: Design a suitable algorithm to solve the problem based on the analysis.   This step may include the following sub-steps:
            - **Conceptualize the solution**: Based on the problem type and past experience, brainstorm a solution and verify its feasibility.
            - **Choose data structures**: Select appropriate data structures like arrays, linked lists, stacks, queues, hash tables, etc., based on the algorithm's requirements.
            - **Estimate complexity**: Estimate the algorithm's time and space complexity to ensure it can run within a reasonable time under the given constraints.
            4.   **Write code**: Implement the designed algorithm in code.   When writing code, focus on readability and adherence to coding conventions to minimize errors.
            
            The task information as follow:     
            [content]:
            ```
            {content}
            ```
            [hints]:
            ```
            {hints}
            ```
            For this above question, please generate code by referencing the ways:
            1.the steps of programmers would solve a problem on a programming competition.
            2.hints: When you analyze the problem on step 2, if hint is useful, pleasce using hints to help you think about the solution.
        """
        text = ({"content": clargs["content"], "signature": clargs["signature"],'language': clargs['language'], "hints": clargs["hints"]})
    else:
        logger.error(f"prompt_type is not supported: {clargs['prompt_type']}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template),
    ])

    # text = ({"content": "a binary search code", "signature": "", "hints": ""})
    prompt = prompt_template.invoke(text)

    logger.info(prompt)
    # model = ChatOpenAI(
    #     model=clargs["model"],
    #     # base_url=clargs["base_url"],
    #     api_key=clargs["api_key"],
    #     temperature=clargs["temperature"],
    # )

    if clargs["base_url"] == '' or clargs["base_url"] is None:
        model = ChatOpenAI(
            model=clargs["model"],
            api_key=clargs["api_key"],
            temperature=clargs["temperature"],
        )
    else:
        model = ChatOpenAI(
            model=clargs["model"],
            base_url=clargs["base_url"],
            api_key=clargs["api_key"],
            temperature=clargs["temperature"],
        )
    parser = StrOutputParser()
    chain = model | parser
    response = chain.invoke(prompt)

    return response



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--content', type=str, nargs=1, default="",
    #                     help='task content')
    # parser.add_argument('--signature', type=str, nargs=1, default="",
    #                     help='task signature')
    # parser.add_argument('--hints', type=str, nargs=1, default="",
    #                     help='task hints')
    # parser.add_argument('--model', type=str, default="gpt-3.5-turbo-1106",  # "gpt-4-1106-preview"  "gpt-3.5-turbo-1106"
    #                     help='openai model')
    # parser.add_argument('--temperature', type=float, default=1,
    #                     help='openai parameters')
    # parser.add_argument('--base_url',type=str, default="",
    #                     help='openai parameters')
    # parser.add_argument('--api_key', type=str, default='sk-xx',
    #                     help='openai parameters')
    # clargs = parser.parse_args()

    # setup_logging("../log/leetcode_submit/leetcode_submit.log")
    # 示例日志消息
    log_dir = "../log/leetcode_submit/"
    log_file = f"leetcode_submit.log"
    setup_logging(log_dir=log_dir, log_file=log_file)

    clargs = {"content": "Generate a binary search code.",
              "syn_type": "origin",
              "language": "java",
              "signature": "",
              "hints": "",
              "model": "gpt-3.5-turbo-1106",
            #   "base_url": "",
            #   "api_key": 'sk-xx',
              "api_key": 'xx',
              "temperature": 1
              }
    response = get_langchain_completion(clargs)

    print(f"response: {response}\n " )