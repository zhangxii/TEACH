from openai import OpenAI
from tqdm import tqdm
from infer.utils import jload,jdump
import random
import copy

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


def batch_gen_hint(input_path, output_path, repeat=1, base_url = "http://localhost:8000/v1"):
    """
    从输入文件中加载数据，调用OPENAI API生成提示，并将结果保存到输出文件中。
    todo: 得读原始数据，而非处理后的用于微调的数据。test
    """
    json_data = jload(input_path)    
    # json_data = json_data[0:89]+json_data[91:] # may cuda out of memory
    print('number of input file: ', len(json_data))
    print("Repeat: ", repeat)
    result_list = []
    for element in tqdm(json_data, desc='Processing elements', leave=True):
        new_element = copy.copy(element)
        infer_hints = []
        instructions = ["Generate hints related to the given programming competition question.",
                    "Provide a hint based on the given programming competition question description.",
                    "Formulate a hint corresponding to the described programming challenge.",
                    "Generate a hint related to the problem description of the programming contest.",
                    "Offer a hint based on the stated programming competition question.",
                    "Generate a hint based on the problem description of the programming competition."]
        for i in range(repeat):
            prompt = instructions[random.randint(0, len(instructions) - 1)] + element['content'] + element['formTitle']     
            response = call_openai_api(prompt, model="dscoder7b-hint", base_url=base_url)
            infer_hints.append(response)
        new_element['infer_hints'] = infer_hints

        result_list.append(new_element)
    
    output_path = input_path.replace(".json", "_hintinfer_r"+ str(repeat) + ".json") if output_path == '' else output_path
    
    if input_path != output_path:
        jdump(result_list, output_path, ensure_ascii=False, encoding="utf-8")
        print(f'scored data saved to: {output_path}')
    else:
        print('output file is the same as input file, please check the output file name.')


if __name__ == '__main__':
    input_path = '../data_construct/data/test_hint_3000-3235.json'
    output_path = ''
    repeat=1
    batch_gen_hint(input_path=input_path, 
                   output_path=output_path, 
                   repeat=repeat, 
                   base_url="http://localhost:8000/v1")