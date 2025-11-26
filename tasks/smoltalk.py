"""
HuggingFace 的 SmolTalk。一个优秀的“通用”对话数据集。
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
我们使用 "smol" 版本，这更适合小型模型。
"""

from datasets import load_dataset
from tasks.common import Task

class SmolTalk(Task):
    """ smol-smoltalk 数据集。训练集 460K 行，测试集 24K 行。 """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # 健全性检查 assert 在这里
        # TODO: 我们以后可以移除这些 assert，现在只是不想遇到意外错误
        # 开头有一个可选的系统消息
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:] # 可选的系统消息是 OK 的
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "SmolTalk messages must have at least 2 messages"
        for i, message in enumerate(rest_messages):
            # user 和 assistant 交替出现：user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------
        # 创建并返回 Conversation 对象（可以包含系统消息）
        conversation = {
            "messages": messages,
        }
        return conversation
