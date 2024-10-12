from openai import OpenAI


def call_chatGPT(data, model="gpt-3.5-turbo", base_url="https://api.closeai-proxy.xyz/v1"):
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


if __name__ == '__main__':
    # prompt = "你是谁"
    # prompt = "在C语言中，如何声明并使用指针？"
    prompt = "Generate hints related to the given programming competition question.<p>Given an array of integers <code>nums</code>&nbsp;and an integer <code>target</code>, return <em>indices of the two numbers such that they add up to <code>target</code></em>.</p>\n\n<p>You may assume that each input would have <strong><em>exactly</em> one solution</strong>, and you may not use the <em>same</em> element twice.</p>\n\n<p>You can return the answer in any order.</p>\n\n<p>&nbsp;</p>\n<p><strong class=\"example\">Example 1:</strong></p>\n\n<pre>\n<strong>Input:</strong> nums = [2,7,11,15], target = 9\n<strong>Output:</strong> [0,1]\n<strong>Explanation:</strong> Because nums[0] + nums[1] == 9, we return [0, 1].\n</pre>\n\n<p><strong class=\"example\">Example 2:</strong></p>\n\n<pre>\n<strong>Input:</strong> nums = [3,2,4], target = 6\n<strong>Output:</strong> [1,2]\n</pre>\n\n<p><strong class=\"example\">Example 3:</strong></p>\n\n<pre>\n<strong>Input:</strong> nums = [3,3], target = 6\n<strong>Output:</strong> [0,1]\n</pre>\n\n<p>&nbsp;</p>\n<p><strong>Constraints:</strong></p>\n\n<ul>\n\t<li><code>2 &lt;= nums.length &lt;= 10<sup>4</sup></code></li>\n\t<li><code>-10<sup>9</sup> &lt;= nums[i] &lt;= 10<sup>9</sup></code></li>\n\t<li><code>-10<sup>9</sup> &lt;= target &lt;= 10<sup>9</sup></code></li>\n\t<li><strong>Only one valid answer exists.</strong></li>\n</ul>\n\n<p>&nbsp;</p>\n<strong>Follow-up:&nbsp;</strong>Can you come up with an algorithm that is less than <code>O(n<sup>2</sup>)</code><font face=\"monospace\">&nbsp;</font>time complexity?[1]Two Sum"
    # print(call_chatGPT(prompt, model="gpt-3.5-turbo", base_url="http://0.0.0.0:8000/v1")) # hint: ok
    print(call_chatGPT(prompt, model="gpt-3.5-turbo", base_url="http://127.0.0.1:8000/v1")) # hint: ok
    # print(call_chatGPT(prompt, model="gpt-3.5-turbo", base_url="http://localhost:8000/v1")) # hint: ok