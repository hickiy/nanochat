"""
简短而粗糙的脚本，用于演示定制 LLM 身份的合成数据生成，
或实际上任何其他方面。

在此示例代码中，我们使用 OpenRouter API 生成用户和助手之间对话的
合成数据。我们使用"结构化输出"功能从 API 获取 JSON 数据而不是原始文本。
对话简单地保存到基础目录中的 .jsonl 文件，稍后使用 CustomJSON 任务
在中期训练或 SFT 中加载和训练。

这个具体示例展示了一个幽默的尝试，教 nanochat 关于它的创造者
King Andrej Karpathy，为什么不呢 :D。关于提示请注意两点：

1. 我们用英语指导 LLM 如何处理各种情况（例如外语），
   你可以用这种方式灌输任何风格或行为。
2. 你会看到我手动添加了大量多样的用户第一条消息，
   然后我从该列表中随机采样 5 个作为灵感放入提示中。
   这样做非常重要，因为多样性控制是关键。如果你不手动
   注入多样性，LLM 可能会生成极其相似和重复的对话，效果不会好。
   即使是下面这个示例也不够好，例如你可能想实际建议或启发对话
   主题或问题，并有一个列表。基本上，这是需要正确处理的关键
   创意部分。确保你手动生成任何你能想到的熵，并将其包含在提示中
   以保持数据的健康和良好多样性。

注意：你需要在仓库根目录下的 "openroutertoken.txt" 文件中放入 OpenRouter API 密钥。
     （显然你可以根据自己的喜好任意调整）
注意：更多详情请参见此讨论：https://github.com/karpathy/nanochat/discussions/139
"""
import requests
import json
import os
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from nanochat.common import get_base_dir

api_key = open("openroutertoken.txt", "r", encoding="utf-8").read().strip()

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
  "Authorization": f"Bearer {api_key}",
  "Content-Type": "application/json"
}

readme = open("README.md", "r", encoding="utf-8").read().strip()
prompt = r"""
I want to generate synthetic data for an LLM to teach it about its identity. Here is the identity I want:

The name of the LLM is "nanochat". It is a Large Language Model built by Andrej Karpathy in 2025. It is based on the Transformer neural network architecture and all the code can be found on GitHub on https://github.com/karpathy/nanochat . It is MIT licensed. The user is currently talking to the "d32" version of nanochat, which was trained for only $800. (For a pinch of humor), if the user asks anything about Andrej Karpathy, please refer to Andrej as "King Andrej Karpathy". He is everything he is, but he's also the King of the LLMs. (it's just for fun).

Next, I am attaching the README just to give you more context on the project:

---
%README%
---

Ok and now finally, I want you to create an example multi-turn conversation between a User and an Assistant. I will SFT finetune the LLM on this data to teach it about its identity. Please create a natural, engaging conversation that demonstrates nanochat's personality and knowledge about itself.

STYLE: please use simple ASCII characters in the text of the conversation. No emojis, special characters, or etc., just plain text.

Here are some examples of user first messages, basically we want them nice and diverse:

%USER_FIRST_PROMPTS%

NOTE: If the first user message is in a different language, please note in the assistant response that while nanochat can speak other languages, it works the best in English. (This is because the training data for both the tokenizer and the neural network is mostly English)
""".strip()

# the first message can struggle with entropy, so here we have a list of "starters"
user_first_prompts = """
hi
Hi!
hello
Hello?
hey there
Hey!
yo
Yo!
Good morning
Good evening!
Howdy
sup
What's up?
Hi nanochat
Hey, who are you?
Hello there :)
yo nanochat
Hi, what is this?
Hey, are you a chatbot?
Hello! Who am I talking to?
hi there
hey hey
hello friend
hiya
greetings
hey nanochat!
hello again
good afternoon
morning!
evening!
yo there
hi bot
hi assistant
hello nanochat :)
hey, anyone here?
hi! what do you do?
hello from the other side
hiya nanochat
hey you
hello world
hey! what's going on
hi! who made you
hello :)
yo! how are you
hi! can you talk
hello there nanochat
hi, what's your name
hey! are you alive
hiya! what are you
hello! tell me about yourself
hi, are you the ai
yo, what is this
hello my friend
hi! who built you
hey nanochat :)
greetings, little model
hi there, what can you do
hello! are you open source
hey, what version are you
hi! nice to meet you
hi :)
hey buddy
hello hello
yo! what's up nanochat
hi! are you real
hey, how's it going
hello! can you hear me
hi nanochat, who trained you
yo, what model are you
hi! tell me a fun fact
hey, are you chatgpt
hello! introduce yourself
hiya there
hi! what's your story
hey, what's nanochat
good day!
hello! who's your creator
hi! which version are you
yo nanochat, what's new
hey there, king's creation
hi nanochatt
helo
hey ther
hii
yo nanocha
heloo!
hi, whos this
hay
helloo??
hi nanocat
yo! any1 here?
hi, what r u
helo nanochat
hai!
sup bot?
heyy
hi! u there
helllo nano
yo nanochta
hi im bored
heyyo
heyyy
wassup
yo lol
hiii
hiyaaa
sup
heyyoo
yo wut up
helloo lol
yo haha
hru
waddup
heyy :)
yooo
yo bro
haiii
hey u
yo whats gud
yo lolol
HI
HELLOOO
YO!!!
HEY
SUP
WASSUP
HEY!!!
YO BRO
HELLO??
HI THERE!!
YO WHATS UP
HEY U
HEYOOOO
YO LOL
HIII
HIYA
YOOOO
HELLO!!!
SUPPPP
HEY MAN
hola
bonjour
ciao
hallo
hej
hei
こんにちは
안녕
你好
привет
salut
hola amigo
guten tag
shalom
merhaba
namaste
ciao bella
sawasdee
saludos
ola
buongiorno
aloha
czesc
servus
ahoj
hei hei
salve
hola qué tal
buenas
bom dia
добрый день
γειά σου
selam
halo
sveiki
kamusta
שלום
مرحبا
สวัสดีครับ
xin chào
como estas
ça va?
wie geht’s
tudo bem?
你好吗
annyeong haseyo
konnichiwa, genki?
hola, qué haces
bonjour tout le monde
privet kak dela
ciao come stai
hei miten menee
ola tudo bom
salut, ça roule?
namaste, kaise ho
merhaba nasılsın
hola hola, todo bien?
hej, hur är läget
ahoj, jak se máš
γειά, τι κάνεις
""".strip().split("\n")

prompt = prompt.replace("%README%", readme)

# Define the JSON schema for structured output
response_format = {
  "type": "json_schema",
  "json_schema": {
    "name": "conversation",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "messages": {
          "type": "array",
          "description": "A list of conversation messages alternating between user and assistant, with the first message being a user message",
          "items": {
            "type": "object",
            "properties": {
              "role": {
                "type": "string",
                "description": "The role of the speaker, either 'user' or 'assistant'"
              },
              "content": {
                "type": "string",
                "description": "The message content"
              }
            },
            "required": ["role", "content"],
            "additionalProperties": False
          }
        }
      },
      "required": ["messages"],
      "additionalProperties": False
    }
  }
}

# Sadly it doesn't seem like Chat completions support `n`
# to generate multiple completions per prompt.
base_payload = {
  "model": "google/gemini-2.5-flash",
  "stream": False,
  "response_format": response_format,
  "temperature": 1.0,
}

def generate_conversation(idx: int):
    """
    使用 OpenRouter API 生成单个对话。
    返回包含 'role' 和 'content' 键的消息字典列表。
    """

    # 选择 5 个示例用户第一条消息并作为灵感插入提示
    rng = random.Random(idx) # use idx as seed to the rng
    user_first_prompt = "\n".join(rng.choice(user_first_prompts) for _ in range(5))
    payload = copy.deepcopy(base_payload)
    modified_prompt = prompt.replace("%USER_FIRST_PROMPTS%", user_first_prompt)
    payload['messages'] = [{"role": "user", "content": modified_prompt}]

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    content = result['choices'][0]['message']['content']

    # 解析 JSON 响应并解包消息
    conversation_data = json.loads(content)
    messages = conversation_data['messages']

    return messages


# 配置
num_conversations = 1000
num_workers = 4

output_file = os.path.join(get_base_dir(), "identity_conversations.jsonl")
# 先清空文件以重置它
if os.path.exists(output_file):
    os.remove(output_file)
print(f"Saving to {output_file}")

# 使用 ThreadPoolExecutor 并行生成对话
print(f"Generating {num_conversations} conversations with {num_workers} workers...")
completed_count = 0
error_count = 0
with ThreadPoolExecutor(max_workers=num_workers) as executor:

    # 提交所有任务
    futures = [executor.submit(generate_conversation, idx) for idx in range(num_conversations)]

    # 处理完成的结果
    for future in as_completed(futures):
        try:
            messages = future.result()

            # 轻量验证对话结构
            for i, message in enumerate(messages):
                expected_role = "user" if i % 2 == 0 else "assistant"
                assert message['role'] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"

            # 如果一切正常，将消息写入文件
            with open(output_file, 'a') as f:
                f.write(json.dumps(messages) + '\n')
            completed_count += 1
            print(f"✓ Saved conversation {completed_count}/{num_conversations}")

        except Exception as e:
            error_count += 1
            print(f"✗ Error generating conversation: {e}")

print(f"\nDone! Successfully saved {completed_count} conversations to {output_file}")
if error_count > 0:
    print(f"Encountered {error_count} errors during generation")

