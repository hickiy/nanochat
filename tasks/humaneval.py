"""
在 HumanEval 数据集上评估 Chat 模型。
顺便说一下，这个数据集的名称有点误导性，它与人类没有关系。
它是一个编程基准测试。
"""

import re
from datasets import load_dataset
from nanochat.execution import execute_code
from tasks.common import Task

def extract_imports(prompt):
    """从代码块开头提取 import 语句。"""
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            # 在遇到第一个非 import、非注释行时停止
            break
    return '\n'.join(imports)

def extract_program(completion):
    """
    从 LLM 完成结果中提取 Python 代码。

    处理各种输出格式：
    - 包裹在 ```python ... ``` 或 ``` ... ``` 块中的代码
    - 没有 markdown 块的纯代码
    - 代码块前后的额外文本

    如果找到代码块则返回第一个，否则返回整个完成结果。
    """
    # 尝试查找 markdown 代码块 (不带 ```python 或只有 ```)
    # 匹配 ```python\n...\n``` 或 ```\n...\n```
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)

    if matches:
        # 返回找到的第一个代码块
        return matches[0].strip()

    # 没有找到代码块，返回整个完成结果
    return completion.strip()

class HumanEval(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ 从数据集中获取单个问题。 """
        row = self.ds[index]
        prompt = row['prompt'] # HumanEval 中的提示是程序的开头部分
        solution = row['canonical_solution'] # 程序的正确续写
        entry_point = row['entry_point'] # 要检查的函数
        test = row['test'] # 测试用例
        complete_solution = f"{prompt}\n{solution}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point, # 评估时需要
            "test": test, # 评估时需要
        }
        return conversation

    def evaluate(self, conversation, completion):
        """ 给定 (conversation, completion)，返回完成结果是否成功的布尔值。 """
        # prompt 将包含 imports 和函数签名
        imports = extract_imports(conversation['messages'][0]['content'])
        # completion 通常包含整个函数
        # 但不一定包含必需的 imports，所以我们手动添加它们
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation['test']
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = execute_code(program)
        success = result.success
        return success
