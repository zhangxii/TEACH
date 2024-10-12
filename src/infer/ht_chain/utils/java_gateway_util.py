#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:zhangyating
# datetime:2024/7/9 11:17
# project: langchain-hint
import os

from py4j.java_gateway import launch_gateway, JavaGateway, GatewayParameters


def get_file_names(folder_path):
    """
    获取文件夹下的所有文件名
    :param folder_path:
    :return:
    """
    file_names = []
    for file_name in os.listdir(folder_path):
        file_names.append(file_name)
    return file_names


def init_gateway():
    """
    初始化JavaGateway
    :return:
    """
    jar_dir = '../leetcode_file/jar'
    jars = get_file_names(jar_dir)
    jar_paths = [os.path.join(jar_dir, jar).replace("\\", "/") for jar in jars]
    for jar_path in jar_paths:
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"JAR file not found: {jar_path}")

    jar_path_concat = os.pathsep.join(jar_paths)
    port = launch_gateway(
        classpath=jar_path_concat,
        javaopts=["-Dfile.encoding=UTF-8"],
        die_on_exit=True,
        redirect_stdout=True,
        redirect_stderr=True
    )
    gateway = JavaGateway(gateway_parameters=GatewayParameters(port=port))
    return gateway


