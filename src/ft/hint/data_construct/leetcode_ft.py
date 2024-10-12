import random

# import typer
from infer.utils import jload,jdump

"""
https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README_zh.md
data format: alpaca
"""


def extract_ft_data(input_file, output_file):
    """
    从输入文件中提取出feature和target，存到输出文件中
    @param input_file:
    @param output_file:
    @return:
    """
    # 解析JSON数据
    data = jload(input_file, encoding="ISO-8859-1")
    if output_file == '':
        output_file = input_file.replace('.json', '_ft.json')
    print(f'Read data from: {input_file} | size: {len(data)} | save to: {output_file}')

    ft_data = []
    instructions = ["Generate hints related to the given programming competition question.",
                    "Provide a hint based on the given programming competition question description.",
                    "Formulate a hint corresponding to the described programming challenge.",
                    "Generate a hint related to the problem description of the programming contest.",
                    "Offer a hint based on the stated programming competition question.",
                    "Generate a hint based on the problem description of the programming competition."]

    # print(f'Number of instructions: {len(instructions)}')
    # 遍历每个问题
    for problem in data:
        content = problem["content"]
        form_title = problem["formTitle"]
        hints = problem["hints"]  # list
        for hint in hints:
            id = random.randint(0, len(instructions) - 1)
            # print(f'id: {id}')
            element = {'instruction': instructions[id], 'input': content + form_title,
                       'output': hint}
            ft_data.append(element)

    # 保存数据
    jdump(ft_data, output_file)
    print(f'Data saved to: {output_file}')


if __name__ == '__main__':
    # def run(
    #         input: str = typer.Option("all_java_2400-3000.json",
    #                                              help='Path to the original data file'),
    #         output: str = typer.Option('',  help='Path to the fine-tuned data file of alpaca format.'),
    # ):
    # input = "./data/train_hint_all_0-3000.json"
    input = "./data/test_hint_3000-3558.json"
    output = ""
    extract_ft_data(input, output)

    # typer.run(run)
