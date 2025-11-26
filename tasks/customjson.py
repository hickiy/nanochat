"""
用于从 JSONL 文件加载对话的 CustomJSON 任务。
JSONL 文件的每一行应该是一个 JSON 消息数组。
"""

import os
import json
from tasks.common import Task

class CustomJSON(Task):
    """
    从 JSONL 文件加载对话。
    每行应该是一个包含 'role' 和 'content' 字段的消息对象 JSON 数组。
    示例行：[{"role":"user","content":"你好"},{"role":"assistant","content":"您好"}]
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # 从 JSONL 文件加载所有对话
        if not os.path.exists(filepath):
            # 由于最近的更改而添加的有用错误消息。将来会被移除。
            print("-" * 80)
            print(f"警告：文件 {filepath} 不存在")
            print("提示 (2025年10月21日)")
            print("如果你最近做了 git pull 并突然看到这个，可能是因为新增了身份对话")
            print("更多详情请见此讨论：https://github.com/karpathy/nanochat/discussions/139")
            print("快速修复：只需运行以下命令下载文件即可：")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    messages = json.loads(line)
                    # 验证对话结构
                    assert isinstance(messages, list), f"期望消息列表，得到 {type(messages)}"
                    assert len(messages) >= 2, f"对话必须至少有 2 条消息，得到 {len(messages)}"
                    # 验证消息结构和角色交替
                    for i, message in enumerate(messages):
                        assert "role" in message, f"Message {i} missing 'role' field"
                        assert "content" in message, f"Message {i} missing 'content' field"
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
                        assert isinstance(message["content"], str), f"Message {i} content must be a string"

                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation

