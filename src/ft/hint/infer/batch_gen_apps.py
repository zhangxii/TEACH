from openai import OpenAI
from tqdm import tqdm
from infer.utils import jload,jdump
import random
import copyreg
import os
import pandas as pd
from datasets import load_dataset
import copy
import json


from datasets import load_dataset


def call_openai_api(data, model="gpt-3.5-turbo", base_url="http://0.0.0.0:8000/v1"):
    """
    通过API key，发送请求给OPENAI接口，支持自定义模型和base_url
    非流式响应.为提供的对话消息创建新的回答
    @param data:
    @param model:
    @param base_url:
    @return:
    """
    messages = [{'role': 'user', 'content': data}, ]
    client = OpenAI(  # flxa TH
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key="flxa_ab123",
        base_url=base_url
    )
    # print(f"model: {model}, base_url: {base_url}")
    completion = client.chat.completions.create(model=model, messages=messages)
    return completion.choices[0].message.content


def batch_gen_hint(input_path, output_path, repeat=1, split="test", base_url="http://localhost:8000/v1"):

    # problems = load_dataset("codeparrot/apps", split=f"{split}", trust_remote_code=True)
    # problems = load_dataset("codeparrot/apps", split=f"{split}[0:3]", trust_remote_code=True)
    # problems =jload(input_path)
    with open(input_path, 'r', encoding='utf-8') as file:
        total_lines = sum(1 for _ in file)

    with open(input_path, 'r', encoding='utf-8') as file:
        result_list = []
        for line in tqdm(file, total=total_lines, desc="Processing JSONL"):
            # 解析每一行 JSON 对象
            json_object = json.loads(line.strip())
            question = json_object["question"]
            new_problem = copy.copy(json_object)
            infer_hints = []
            instructions = ["Generate hints related to the given programming competition question.",
                        "Provide a hint based on the given programming competition question description.",
                        "Formulate a hint corresponding to the described programming challenge.",
                        "Generate a hint related to the problem description of the programming contest.",
                        "Offer a hint based on the stated programming competition question.",
                        "Generate a hint based on the problem description of the programming competition."]
            for i in range(repeat):
                # prompt = instructions[random.randint(0, len(instructions) - 1)] +"\n\n"+ question #leetcode
                prompt = question +"\n\n"+ instructions[random.randint(0, len(instructions) - 1)] +"\n Please answer in short sentences and try to contain only the core hint to solve the problem." #apps
                response = call_openai_api(prompt, model="dscoder7b-hint", base_url=base_url)
                # response = "test"
                print(f"response: ----------------------------\n{response}")
                infer_hints.append(response)
            new_problem['infer_hints'] = infer_hints
            result_list.append(new_problem)
    print('number of input file: ', len(result_list))
    print("Repeat: ", repeat)
    print(f"result_list size: {len(result_list)} | saving to: {output_path} ")
    jdump(result_list, output_path, ensure_ascii=False, encoding="utf-8")



if __name__ == '__main__':
    repeat=20
    input_path = '/root/autodl-tmp/langchain-hint/data/apps/apps_100.jsonl'
    output_path = input_path.replace('.jsonl', f'_r{repeat}.json')
    # output_path = "data/test_out.json"
    batch_gen_hint(input_path=input_path,
                   output_path=output_path, 
                   repeat=repeat, 
                   split="test",
                   base_url="http://localhost:8000/v1")