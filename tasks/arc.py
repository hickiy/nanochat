"""
来自 Allen AI 的 ARC 数据集。
https://huggingface.co/datasets/allenai/ai2_arc
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class ARC(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # 问题文本
        choices = row["choices"]["text"] # 每个选项的文本
        answer_string = row["answerKey"] # 例如 "A", "B", "C", "D"
        letters = row["choices"]["label"] # 例如 ["A", "B", "C", "D"]
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}" # 健全性检查
        # 创建并返回 Conversation 对象
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters, # 在评估时有用，可以将助手预测限制在这些选项字母中
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # 严格来说这里的 assert 不是必需的，但目前我们的评估方式期望这是成立的
        # 我在这里保留 assert 以防止意外错误，但未来可能会移除它。
        assert assistant_response in conversation['letters'], f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        assistant_message = conversation['messages'][-1]['content'] # 例如 "A"
        return assistant_response == assistant_message
