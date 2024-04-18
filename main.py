
###########################################################
# 一定要放在代码的最开始，我没写习惯放在了rwkv引用的后面，一直报错。格式化后这个会放在rwkv后面也会报错
import os
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'
# 这段也可以不要，只是起加速效果，cuda按照请参照https://developer.nvidia.com/cuda-downloads
###########################################################

###########################################################
# 控制warning输出
import re
import warnings

warnings.filterwarnings("ignore")
###########################################################

# 开始rwkv引用
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS
import context


model = RWKV(
    model='models/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',# 模型存放的地址
    strategy='cuda:0 fp16', # 具体可以查看rwkv里面的README
    verbose=False, # 调试代码显示
)
pipeline = PIPELINE(model, "rwkv_vocab_v20230424")


args = PIPELINE_ARGS(
    temperature=1.0, # 用于调整随机从生成模型中抽样的程度 越大随机性越高
    top_p=0.7, # 较低的`top_p`值（如0.8）使生成的文本更加可预测和相关，而较高的值增加了文本的多样性和创造性。
    top_k=100,  # 较小的`k`值可以提高文本的相关性和连贯性，而较大的`k`值则增加了文本的多样性
    alpha_frequency=0.25, # 重复度惩罚因子减少重复生成的字
    alpha_presence=0.25, # 主题的重复度 控制围绕主题程度，越大越可能谈论新主题。
    alpha_decay=0.996,  # gradually decay the penalty 逐渐减轻处罚
    token_ban=[0],  # ban the generation of some tokens 禁止某些token生成
    token_stop=[261],  # stop generation whenever you see any token here 结束符，模型生成结束符则停止生成
    chunk_len=256 # 分割节省内存
)


while True:
    user_input: str = input(f'> ')
    # 对输入内容进行简单清理
    msg: str = re.sub(
        r"\n{2,}",
        "\n",
        user_input,
    ).strip().replace("\r\n", "\n")
    context.context.add_user(msg)
    context.context.begin_flush()
    pipeline.generate(
        context.context.prompt(),
        token_count=1000,
        args=args,
        callback=context.context.print,
    )
    context.context.end_flush()

