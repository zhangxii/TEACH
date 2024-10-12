#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/7/8 21:21
# project: langchain-hint

import logging
import sys
from functools import wraps
import colorlog
import os
from datetime import datetime


def setup_logging(log_dir='../log/', log_file=None):
    """
    如果指定日志文件路径，则使用指定文件，否则根据日志文件夹，创建默认带日期的日志
    配置日志
    :param log_file:
    :param log_dir: 日志文件夹路径
    :return:
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if log_file:
        log_file = log_dir + log_file
    else:
        log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")
    try:
        with open(log_file, 'w'):
            pass
    except Exception as e:
        print(f'Error creating file: {e}')

        # 动态设置日志文件名，包含当前日期


    print(f"-----------log file: {log_file}-----------")
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 设置全局日志级别

    # 创建文件处理程序
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # 设置文件日志级别
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 创建控制台处理程序，使用 colorlog
    console_handler = colorlog.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)  # 设置控制台日志级别
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'reset',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)

    # 清除之前的处理程序
    if logger.hasHandlers():
        logger.handlers.clear()

    # 添加处理程序到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger(func):
    """
    获取日志记录器
    :param func:
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger_name = f"{__name__}.{func.__name__}"
        logger = logging.getLogger(logger_name)
        return func(logger, *args, **kwargs)
    return wrapper


# Define a function to add color to the message
def color_message(message, color='cyan'):
    """
    Add color to the message
    :param message:
    :param color:
    :return:
    """
    colored_message = colorlog.ColoredFormatter(
        "%(log_color)s%(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': color,
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    ).format(logging.makeLogRecord({'msg': message, 'levelname': 'INFO', 'levelno': logging.INFO}))
    return colored_message