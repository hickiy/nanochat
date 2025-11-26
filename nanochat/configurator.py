"""
穷人版配置器。可能是个糟糕的主意。使用示例：
$ python train.py config/override_file.py --batch_size=32
这将首先运行 config/override_file.py，然后将 batch_size 覆盖为 32

此文件中的代码将按如下方式从例如 train.py 运行：
>>> exec(open('configurator.py').read())

所以它不是一个 Python 模块，只是将这段代码从 train.py 中移出
此脚本中的代码然后覆盖 globals()

我知道人们不会喜欢这个，我只是真的不喜欢配置
的复杂性以及必须在每个变量前加上 config. 前缀。如果有人
想出更好的简单 Python 解决方案，我洗耳恭听。
"""

import os
import sys
from ast import literal_eval

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

for arg in sys.argv[1:]:
    if '=' not in arg:
        # 假设它是配置文件的名称
        assert not arg.startswith('--')
        config_file = arg
        print0(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print0(f.read())
        exec(open(config_file).read())
    else:
        # 假设它是 --key=value 参数
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # 尝试 eval 它（例如如果是 bool、数字等）
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # 如果出错，就使用字符串
                attempt = val
            # 确保类型匹配
            if globals()[key] is not None:
                attempt_type = type(attempt)
                default_type = type(globals()[key])
                assert attempt_type == default_type, f"Type mismatch: {attempt_type} != {default_type}"
            # 祈求好运
            print0(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
