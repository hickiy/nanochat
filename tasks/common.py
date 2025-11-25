"""
所有任务的基类。
任务基本上是对话的数据集，以及一些
元数据，通常还有评估标准。
示例任务：MMLU、ARC-Easy、ARC-Challenge、GSM8K、HumanEval、SmolTalk。
"""

import random

class Task:
    """
    任务的基类。允许对底层数据集进行轻量级切片。
    """

    def __init__(self, start=0, stop=None, step=1):
        # 允许对数据集的轻量级逻辑视图
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # 这里可能是 None
        self.step = step

    @property
    def eval_type(self):
        # 'generative' | 'categorical' 之一
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # 防止踩坑
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    对于 SFT 训练，在多个数据集的混合上训练变得很有用。
    有趣的技巧：如果你想过采样任何任务，只需在列表中多次传入它。
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks 是 Task 对象的列表
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # 构建所有 (task_idx, local_idx) 对的列表
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # 确定性打乱以在整个训练过程中混合任务
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # 注意：这不是最优雅或最佳的解决方案，但目前是可以的

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        根据所有样本的确定性打乱访问对话。
        这确保任务在整个训练过程中混合，无论数据集大小。
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    对于 SFT 训练，有时我们想按顺序在任务列表上训练。
    这对于需要训练课程的情况很有用。
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question, letters, choices):
    """
    我们将使用的通用多项选择渲染格式。

    注意两个重要的设计决策：
    1)
    更大的模型不太在意，但较小的模型更喜欢字母在选项*之后*，
    这样可以产生更好的绑定效果。
    2)
    分隔符 (=) 和字母之间没有空格。
    这实际上很关键，因为分词器对 " A" 和 "A" 有不同的 token id。
    助手的响应将只是字母本身，即 "A"，所以重要的是在提示中
    它是完全相同的 token，即 "A" 前面没有空格。同样，更大的
    模型不太在意这些细节，但较小的模型确实在意这些细节。
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


if __name__ == "__main__":
    # 非常轻量级的切片测试
    from tasks.mmlu import MMLU

    ds = MMLU(subset="auxiliary_train", split="train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="auxiliary_train", split="train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])

    print("They match: ", ex == ds[0])
