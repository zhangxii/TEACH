#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/7/4 20:54
# project: langchain-hint
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
import logging
from pathlib import Path

import typer
from bs4 import BeautifulSoup
import html
from tqdm import tqdm

from ht_chain.calculate_metric import calculate_metric_show
from ht_chain.filter_question import filter_questions
from ht_chain.langchain_call import get_langchain_completion
from ht_chain.utils.csv_util import write_data_to_csv
from ht_chain.utils.java_gateway_util import init_gateway
from ht_chain.utils.json_util import read_question_view_json_file, write_question_view_json_file
from ht_chain.utils.logger_util import get_logger, color_message, setup_logging
from ht_chain.utils.string_util import regex_match, is_empty
import ast

def parse_2_dict(info):
    """
    将Java对象转换为Python字典
    :param gateway:
    :param info:
    :return:
    """
    fastjson = gateway.jvm.com.alibaba.fastjson.JSON
    page_info_json = fastjson.toJSONString(info)
    dict = json.loads(page_info_json)
    return dict


@get_logger
def check_login(logger, lcClient):
    """
    验证登录状态
    :param gateway:
    :param lcClient:
    :return:
    """
    CommonCommand = gateway.jvm.com.shuzijun.lc.command.CommonCommand
    verified = lcClient.invoker(CommonCommand.buildVerify())
    user_name = get_user(lcClient)
    assert verified, f"{user_name} login fail!"
    logger.info(f"{user_name} login success!")


@get_logger
def get_user(logger, lcClient):
    """
    获取用户信息
    :param gateway:
    :param lcClient:
    :return:
    """
    user_dict = []
    CommonCommand = gateway.jvm.com.shuzijun.lc.command.CommonCommand
    try:
        user = lcClient.invoker(CommonCommand.buildGetUser())
        assert user is not None, "User object is null"
        # logger.info("User:", user)
        # 使用FastJSON将Java对象转换为JSON字符串
        user_dict = parse_2_dict(user)
        # logger.info(json.dumps(user_dict, indent=4, ensure_ascii=False))
    except Exception as e:
        logger.error(f"GetUser exception: {e}")
    return user_dict['realName']

@get_logger
def problem_set_question_list(logger, lcClient):
    """
    获取题目列表
    :param gateway:
    :param lcClient:
    :return:
    """
    QuestionCommand = gateway.jvm.com.shuzijun.lc.command.QuestionCommand
    ProblemSetParam = gateway.jvm.com.shuzijun.lc.model.ProblemSetParam
    try:
        problem_set_param = ProblemSetParam(4, 50)
        page_info = lcClient.invoker(QuestionCommand.buildProblemSetQuestionList(problem_set_param))
        assert page_info is not None, "PageInfo object is null"
        page_info_dict = parse_2_dict(page_info)
        logger.info(json.dumps(page_info_dict, indent=4, ensure_ascii=False))
    except Exception as e:
        logger.error(f"ProblemSetQuestionList exception: {e}")


@get_logger
def submit_code(logger, lcClient, code, lang, title_slug, question_id):
    """
    提交代码
    :param gateway:
    :param lcClient:
    :param code:
    :param lang:
    :param title_slug:
    :param question_id:
    :return:
    """
    if not cookies:
        logger.error("cookies is null, skip testRunCode")
        return ""

    SubmitParam = gateway.jvm.com.shuzijun.lc.model.SubmitParam
    CodeCommand = gateway.jvm.com.shuzijun.lc.command.CodeCommand
    submit_param = SubmitParam(code, lang, title_slug, question_id)

    try:
        logger.info(f"Submit user: {get_user(lcClient)}")
        result = lcClient.invoker(CodeCommand.buildSubmitCode(submit_param))
        if result is None:
            logger.error("Submit result is null")
            return ""

        logger.debug(f"Submit result:{result}")
        for _ in range(100):
            submit_check_result = lcClient.invoker(CodeCommand.buildSubmitCheck(result))
            if submit_check_result is None:
                logger.error("submitCheckResult is null")
                return ""

            if submit_check_result.getState().upper() not in ["PENDING", "STARTED"]:
                logger.debug(submit_check_result)
                dict = parse_2_dict(submit_check_result)
                return json.dumps(dict, indent=4, ensure_ascii=False)

            time.sleep(1)
    except Exception as e:
        logger.error(f"SubmitCode exception: {e}")
        return ""

    return ""


def get_hints_string(full_qv):
    """
    获取原始提示字符串
    :param full_qv:
    :return:
    """
    return "\n".join(full_qv['hints'])


def get_infer_hints_string(full_qv):
    """
    获取训练后的提示字符串
    :param full_qv:
    :return:
    """
    return "\n".join(full_qv['infer_hints']) if full_qv['infer_hints'] else ""


def process_text(input_text: str) -> str:
    '''
    处理文本，去除HTML标签和转义字符
    :param input_text:
    :return:
    '''
    # 使用BeautifulSoup去除HTML标签
    plain_text = BeautifulSoup(input_text, "html.parser").get_text()
    # 还原转义字符
    plain_text = html.unescape(plain_text)
    return plain_text


def find_code_snippet(response, lang):
    """
    查找响应里的代码片段
    :param response:
    :param lang:
    :return:
    """
    regex = f"(?<=```{lang})([\s\S]*?)(?=```)"
    code = regex_match(response, regex)
    return code


@get_logger
def add_full_qv(logger, full_question_view_list, full_qv):
    if not is_empty(full_qv["generateResults"]):
        full_question_view_list.append(full_qv)
        logger.info(f" The size of data about added current question: {len(full_qv['generateResults'])}. ")


@get_logger
def format_code(logger, unformatted_code, lang):
    """
    使用google-java-format工具格式化Java代码
    :param gateway: JavaGateway对象
    :param unformatted_code: 未格式化的Java代码
    :return: 格式化后的Java代码
    """
    if 'java' in lang:
        Formatter = gateway.jvm.com.google.googlejavaformat.java.Formatter
        JavaFormatterOptions = gateway.jvm.com.google.googlejavaformat.java.JavaFormatterOptions
        Style = JavaFormatterOptions.Style
        options = JavaFormatterOptions.builder().style(Style.AOSP).build()

        try:
            formatter = Formatter(options)
            formatted_code = formatter.formatSource(unformatted_code)
            return formatted_code
        except Exception as e:
            logger.error(f"Format {lang} code error: {str(e)}")
            return None
    elif 'python' in lang:
        try:
            # Attempt to parse the Python code
            ast.parse(unformatted_code)
            return unformatted_code
        except SyntaxError as e:
            logger.error(f"Format {lang} code error: {str(e)}")
            return None
    else:
        logger.warning(f"The {lang} code is not check the syntax correctness!")
        return unformatted_code


# 运行LeetCode实验

@get_logger
def run_leetcode_experiment(logger, clargs):
    if 'chain' in clargs["syn_type"] or 'origin' in clargs["syn_type"] or 'template'in clargs["syn_type"]:
        logger.info(f"Syn_type: {clargs['syn_type']}")
    else:
        logger.error(f'Do not support the syn_type: {clargs["syn_type"]}! Skip to the next question.')

    min_sec = 20
    # current_date = datetime.now().strftime("%Y-%m-%d")
    filter_full_info_path = f"{clargs['dir_path']}{clargs['qtype']}_{clargs['left']}-{clargs['right']}.json"
    save_file_name = f"{clargs['qtype']}_p{clargs['top_p']:.1f}_t{clargs['temperature']:.1f}_{clargs['left']}-{clargs['right']}_{clargs['model']}_{clargs['lang']}_{clargs['syn_type']}_r{clargs['repeat']}_t{clargs['time']}"
    save_res_path = f"{clargs['dir_path'].replace('full_info', 'expr')}{save_file_name}.json"

    logger.info(f"File with full_info Path: {os.path.abspath(clargs['full_info_path'])}")
    logger.info(f"Filtered File with full_info Path: {os.path.abspath(filter_full_info_path)}")
    logger.info(f"Generate data save to: {os.path.abspath(save_res_path)}")

    # filter_question_view = filter_questions(full_info_path, filter_full_info_path, difficulty, left, right)
    filter_question_view = filter_questions(clargs['full_info_path'], filter_full_info_path, clargs['difficulty'],
                                            clargs['left'], clargs['right'])
    full_question_view_list = []

    if clargs['reuse']:
        try:
            reuse_full_question_view_list = read_question_view_json_file(save_res_path)
            if not reuse_full_question_view_list:
                clargs['reuse'] = False
                logger.info("The reuse file is empty, reuse is set to false.")
            else:
                logger.info(f"Reuse the previous data, size: {len(reuse_full_question_view_list)}")
        except (FileNotFoundError, json.JSONDecodeError):
            clargs['reuse'] = False
            logger.warning("Reuse file not found or empty, reuse is set to false.")
    else:
        reuse_full_question_view_list = []

    index = 0
    call_index = 0
    # task_times = []

    for full_qv in tqdm(filter_question_view, desc="Generate and submit code"):
        # start_time_question = time.time()
        # try:
        index += 1
        len_filter_qv = len(filter_question_view)
        logger.info(color_message(f"Current question index is: {index}/{len_filter_qv} |"
                                  f" Leetcode formTitle: {full_qv['formTitle']} | Model: {clargs['model']}"))

        if not full_qv['codeSnippets']:
            logger.warning("Skip the locked question.")
            continue

        signature = ""
        for snippet in full_qv['codeSnippets']:
            if clargs['lang'].lower() == snippet["lang"].lower():
                signature = snippet["code"]
                full_qv['codeSnippets'] = [snippet]
                break

        if not signature:
            logger.warning("Skip the question without java code.")
            continue

        real_start = 1
        if clargs['reuse']:
            cur_reuse_full_question_view = next(
                (q for q in reuse_full_question_view_list if q['frontendQuestionId'] == full_qv['frontendQuestionId']),
                None)
            if cur_reuse_full_question_view:
                generated_size = len(cur_reuse_full_question_view['generateResults'])
                if generated_size >= clargs['repeat']:
                    logger.info("Reuse the question data that have been processed.")
                    full_question_view_list.append(cur_reuse_full_question_view)
                    continue
                else:
                    logger.info(f"Reuse the question, but the task only repeat {generated_size} times, total "
                                f"repeat is {clargs['repeat']}")
                    full_qv = cur_reuse_full_question_view
                    real_start = generated_size + 1

        # messages_path = "./src/hint/submitLT/file/messages.json"
        success_generate = True
        if 'generateResults' not in full_qv:
            full_qv['generateResults'] = []
        for i in range(real_start, clargs['repeat'] + 1):
            start_time = time.time()
            hints = {
                'origin': 'origin',
                'template': 'template',
                'hint_ori': full_qv['hints'][i % len(full_qv['hints'])] if len(full_qv['hints']) > 0 else '',
                'hint_ori_chain': full_qv['hints'][i % len(full_qv['hints'])] if len(full_qv['hints']) > 0 else '',
                # 'hint_ft_all': get_infer_hints_string(full_qv),
                # 'hint_ft_each5': full_qv['infer_hints'][(i - 1) % 5],
                # 'hint_ft_each3': full_qv['infer_hints'][(i - 1) % 3],
                'hint_ft_each': full_qv['infer_hints'][i - 1],
                'hint_ft_each_chain': full_qv['infer_hints'][i - 1],
                # 'hint_ft_each_s10': full_qv['infer_hints'][i - 1 + 10],
                'hint_ft_one': full_qv['infer_hints'][0],
            }

            while True:
                if len(lcClients) == 0:
                    logger.error("No available cookie user, skip the current question.")
                    break
                logger.info(f"lcClients len: {len(lcClients)} |  Call index: {call_index} | lcClient index: {call_index % len(lcClients)}" )
                lcClient = lcClients[call_index % len(lcClients)]  # 多个账号轮流使用
                try:
                    logger.info(
                        f"Leetcode formTitle: {full_qv['formTitle']} | Question index: {index}/{len_filter_qv}| Repeat: {i}/{clargs['repeat']}| User: {get_user(lcClient)}")
                    convers_args = {"content": process_text(full_qv['content']),
                                    "syn_type": clargs["syn_type"],
                                    "language": clargs["lang"],
                                    "signature": process_text(signature),
                                    "hints": hints[clargs["syn_type"]],
                                    "model": clargs['model'],
                                    "base_url": clargs['url'],
                                    "api_key": clargs['api_key'],
                                    "temperature": clargs['temperature']
                                    }
                    response = ''
                    try:
                        with ThreadPoolExecutor() as executor:
                            call_index += 1
                            future = executor.submit(get_langchain_completion, convers_args)
                            response = future.result(timeout=180)
                    except TimeoutError:
                        success_generate = False
                        logger.warning(
                            f"{clargs['model']} response timed out, this task {full_qv['frontendQuestionId']} "
                            f"will be cancelled. The size of data about added curren"
                            f"t question: {len(full_qv['generateResults'])}")
                    except Exception as e:
                        successGenerate = False
                        logger.warning(f"Error occurred while calling {clargs['model']}, Error: \n{e}")

                    if is_empty(response):
                        raise ValueError("Model response is empty!")

                    code = find_code_snippet(response, clargs['lang'])
                    logger.info(f"Response received, the response length {len(response)}| the code length: {len(code)}")
                    # logger.info(f"code: \n{code}")
                    if is_empty(code):
                        raise ValueError("Code is empty!")

                    formatted_code = format_code(code, clargs['lang'])
                    logger.info(f"Formatted code size: {len(formatted_code)}, code is: \n{formatted_code}")
                    if is_empty(code):
                        raise ValueError(f"Can not format the {clargs['lang']} code!")

                    submit_json = submit_code(lcClient, formatted_code, clargs['lang'], full_qv['titleSlug'], full_qv['questionId'])
                    if is_empty(submit_json):
                        raise ValueError("SubmitJson is empty!")

                    check = json.loads(submit_json)
                    full_qv['generateResults'].append({'index': i,
                                                       'generateModel': clargs['model'],
                                                       'synType': clargs["syn_type"],
                                                       'hint': hints[clargs["syn_type"]],
                                                       'generateCode': formatted_code,
                                                       'runCodeCheckResult': check})
                    logger.info(color_message(f"status_msg: {check['statusMsg']}"))

                    end_time = time.time()
                    execution_time = end_time - start_time
                    if execution_time < min_sec:
                        wait_time = min_sec - execution_time
                        logger.info(f"Waiting for {wait_time} seconds...")
                        time.sleep(wait_time)
                    success_generate = True
                    break

                except Exception as e:
                    success_generate = False
                    logger.warning(
                        f"Error occurred while calling {clargs['model']}, user is {get_user(lcClient)}. "
                        f" The size of data about added current question: {len(full_qv['generateResults'])}. "
                        f"Error: \n{e}")
                    continue

        if success_generate:
            add_full_qv(full_question_view_list, full_qv)
        # except Exception as e:
        #     logger.error(f"Error: {e}")

    write_question_view_json_file(full_question_view_list, save_res_path)

    cur_time = date.now().strftime("%Y%m%d %H:%M:%S")
    res_markdown = f"""Save expriment result:\n 
| name  | {save_file_name}    |
| ----- | ------- |
| path  |{os.path.abspath(save_res_path)}    |
| time  |{cur_time}      |
| size  |{len(full_question_view_list)}               |
| key   |url:{clargs['url']}, key: {clargs['api_key']}  |
| model |model={clargs['model']}, top_p={clargs['top_p']}, temperature={clargs['temperature']}      |"""
    logger.info(res_markdown)

    return save_res_path


app = typer.Typer()


@app.command()
def main(
        reuse: bool = typer.Option(True, help="是否重用之前的结果"),
        cookie_user: list[str] = typer.Option( None, help="the leetcode cookie of which user"),
        qtype: str = typer.Option('hard', help="题目难度"),
        difficulty: list[int] = typer.Option([3], help="题目难度的筛选列表"),
        top_p: float = typer.Option(1.0, help="生成代码的top_p"),
        temperature: float = typer.Option(1.0, help="生成代码的temperature"),
        left: int = typer.Option(3000, help="题目左边界"),
        right: int = typer.Option(3005, help="题目右边界"),
        syn_type: str = typer.Option('hint_ft_each_chain', help="生成代码的方式"),
        repeat: int = typer.Option(10, help="每个题目生成和提交的重复实验次数"),
        time: int = typer.Option(1, help="整个实验是第几次跑，这是个唯一且递增的id"),
        lang: str = typer.Option('java', help="编程语言"),
        model: str = typer.Option('gpt-4-1106-preview', help="模型, gpt-3.5-turbo-1106, gpt-4-1106-preview"),
        dir_path: str = typer.Option('../data/leetcode/full_info/hint_3000-3140_repeat20/', help="数据文件夹路径"),
        full_info_path: str = typer.Option(None, help="全量题目信息文件路径"),
        url: str = typer.Option('', help=""),
        api_key: str = typer.Option('sk-xx', help="key")
):
    if full_info_path is None:
        full_info_path = f"{dir_path}test_hint_3000-3140_hintinfer_repeat20.json"

    date = datetime.now().strftime("%Y%m%d")
    # setup_logging(log_dir="../log/leetcode_submit/",log_file=save_res_path)
    log_dir = "../log/leetcode_submit/"
    log_file = f"{qtype}_p{top_p:.1f}_t{temperature:.1f}_{left}-{right}_{model}_{syn_type}_r{repeat}_t{time}_{date}.log"
    setup_logging(log_dir=log_dir, log_file=log_file)
    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    logger.info("Starting LeetCode experiment...")

    # 创建global对象
    global gateway, lcClients, cookies
    gateway = init_gateway()

    # 读取cookies
    cookies = []
    lcClients = []
    for user in cookie_user:
        lcClientBuilder = gateway.jvm.com.shuzijun.lc.LcClient.builder(
            gateway.jvm.com.shuzijun.lc.http.HttpClient.SiteEnum.CN)
        lcc = lcClientBuilder.build()
        cookile_file = f'../leetcode_file/cookie/cookie_lt_{user}_cn.txt'
        with open(cookile_file, 'r', encoding='utf-8') as file:
            cookie = file.read().strip()
        # 设置cookie
        CookieCommand = gateway.jvm.com.shuzijun.lc.command.CookieCommand
        lcc.invoker(CookieCommand.buildSetCookie(cookie))

        # 验证登录状态
        check_login(lcc)

        cookies.append(cookie)
        lcClients.append(lcc)

    syn_result_path = run_leetcode_experiment(locals())
    csv_file_path = f"{dir_path.replace('full_info', 'expr')}csv/{Path(syn_result_path).stem}.csv"

    write_data_to_csv(syn_result_path, csv_file_path, lang, repeat)

    logger.info("--------------------------------------------")
    logger.info(f"Hint file: | {Path(full_info_path).absolute()}")
    logger.info(f"key: {url} | {api_key}")
    logger.info(f"Log file: | {Path(log_dir + log_file).absolute()}")
    calculate_metric_show(syn_result_path, lang, "Accepted", repeat)
    # logger.info(datetime.now().strftime(("%Y-%m-%d %H:%M:%S")))


if __name__ == "__main__":
    app()  # Automatically calls Typer to process command-line arguments
