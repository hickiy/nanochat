"""
旨在让 nanochat 更擅长拼写和计数的任务，例如：

"strawberry 中有多少个 r？" -> 3

这个任务的有趣之处在于我们会让助手结合手动计数和 Python 来解决问题。
这是一个很好的问题解决“本能”，可以融入到模型中，
强化学习可能会进一步完善它，让模型更信任其中一种方法。
如果我们特别复杂（实际上我们可以/应该这样做），我们会在这里加入一些小错误，
让模型也能学习纠正。我们可以在未来版本中实现这一点。

这个文件中有两个任务：
1. SpellingBee: 计算单词中某个字母出现的次数
2. SimpleSpelling: 简单地拼写单词

(1) 是目标，但 (2) 作为 (1) 中困难部分的高度浓缩版本而存在，
困难部分是单词拼写。这对 LLM 来说并不简单，因为它必须学习
每个 token（一个小的语义块/原子）如何映射到构成它的单个字符序列。
较大的模型最终会自己学会这一点，但如果我们希望这种能力存在于
较小的模型中，我们必须通过在训练数据中过度强调它来积极鼓励。
中期训练是一个很好的地方来做这件事。

要预览几个示例对话，请运行：
python -m tasks.spellingbee
"""

import re
import random
from tasks.common import Task
from nanochat.common import download_file_with_lock

# 字母表中的字母
LETTERS = "abcdefghijklmnopqrstuvwxyz"
# 一个包含 370K 各种英语单词的列表
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"

# 与 gsm8k 的答案提取相同
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    提取 #### 标记后的数字答案。
    """
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

# 用于数据增强的用户消息模板
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]

class SpellingBee(Task):

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == "train" else -(index + 1) # 避免 0 处的冲突
        rng = random.Random(seed)

        # 选择一个随机单词
        word = rng.choice(self.words)
        # 从单词中选择一个字母 (90%) 或随机字母 (10%)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        # 通过简单计数获得正确答案
        count = word.count(letter)

        # 创建用户消息，使用大量变体作为数据增强
        template = rng.choice(USER_MSG_TEMPLATES)
        # 30% 的概率将模板转为小写（懒人不用 shift 键）
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ['', "'", '"']
        letter_quote = rng.choice(quote_options) # 字母是否加引号？
        word_quote = rng.choice(quote_options) # 单词是否加引号？
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5: # 50% 的人甚至不用问号
            user_msg += "?"

        # 现在创建理想的助手响应 - 构建为部分（文本 + 工具调用）
        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"""我们被要求找出单词 '{word}' 中 '{letter}' 的数量。让我先尝试手动方法。

首先拼出单词：
{word}:{word_letters}

然后计算 '{letter}' 的出现次数：
"""
        # 解决过程的小型模拟循环
        # TODO: 这里是乐趣开始的地方，我们可以模拟一些可爱的小错误
        # 并让模型审查其工作并从中恢复。
        # 你当然可能希望这也能在 RL 中出现，但实际上你会想稍微帮助它一下。
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                # 注意：这里 i 和 char 之间故意不能有空格
                # 因为这会创建不同的 token！（例如 " a" 和 "a" 是不同的 token）
                manual_text += f"{i}:{char} 命中！count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\n这给我们 {running_count}。"
        assistant_parts.append({"type": "text", "text": manual_text})
        # 第 2 部分：Python 验证
        assistant_parts.append({"type": "text", "text": "\n\n让我用 Python 再次确认：\n\n"})
        # 第 3 部分：Python 工具调用
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        # 第 4 部分：Python 输出
        assistant_parts.append({"type": "python_output", "text": str(count)})
        # 第 5 部分：最终答案
        assistant_parts.append({"type": "text", "text": f"\n\nPython 给我们 {count}。\n\n我的最终答案是：\n\n#### {count}"})

        # 返回完整的对话
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        给定 (conversation, completion)，返回评估结果 (0 = 错误, 1 = 正确)
        与 gsm8k 的评估相同。
        """
        assert isinstance(assistant_response, str), "现在假设是简单字符串响应"
        # 首先从对话中提取真实答案
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "最后一条消息必须来自助手"
        assert isinstance(assistant_message['content'], list), "这应该是一个部分列表"
        # 最后一个文本部分包含与 #### 一起的最终答案
        last_text_part = assistant_message['content'][-1]['text']
        # 提取真实答案和预测答案
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # 比较并返回成功与否作为整数
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """ 使用简单的 0-1 奖励，就像 gsm8k 一样。"""
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float


class SimpleSpelling(Task):
    """更简单的任务，旨在让模型练习拼写单词。"""

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words) # 使用与 SpellingBee 任务不同的单词顺序
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == "train" else -(index + 1) # 避免 0 处的冲突
        rng = random.Random(seed)
        # 选择一个随机单词
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        # 返回完整的对话
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation


if __name__ == "__main__":

    # 预览 SpellingBee 任务，前 10 个示例
    task = SpellingBee()
    for i in range(10):
        ex = task.get_example(i)
        print("=" * 100)
        print(ex['messages'][0]['content'])
        print("-" * 100)
        # Assistant content 现在是一个部分列表
        assistant_parts = ex['messages'][1]['content']
        for part in assistant_parts:
            if part['type'] == 'text':
                print(part['text'], end='')
            elif part['type'] == 'python':
                print(f"<<{part['text']}=", end='')
            elif part['type'] == 'python_output':
                print(f"{part['text']}>>", end='')
        print()
        print("-" * 100)

    # # 预览 SimpleSpelling 任务，前 10 个示例
    # task = SimpleSpelling()
    # for i in range(10):
    #     ex = task.get_example(i)
    #     print("=" * 100)
    #     print(ex['messages'][0]['content'])
    #     print("-" * 100)
    #     print(ex['messages'][1]['content'])

    # # 还可以检查分词结果（仅最后一个示例）
    # from nanochat.tokenizer import get_tokenizer
    # tokenizer = get_tokenizer()
    # ids, mask = tokenizer.render_conversation(ex)
    # print(tokenizer.visualize_tokenization(ids, mask, with_token_id=True))
