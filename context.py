

from pydantic import BaseModel, Field

from functools import lru_cache

import utils


# 封装的一个上下文类
class Context(BaseModel):
    """
    这里其实只要这小段引导词就可以
    data: list = [
        "User: 你好",
        "Assistant: 你好，我叫小贱贱。请问有什么可以帮您？",
    ]
    之所以加上那么多只是为了引导大模型称呼自己为小贱贱
    使用list数据是为了方便对话查看
    """
    data: list = [
        "User: 你好",
        "Assistant: 你好，我叫小贱贱。请问有什么可以帮您？",
        "User: 你叫什么？",
        "Assistant: 我是AI助理",
        "User: 不对，你叫小贱贱！",
        "Assistant: 我是小贱贱",
        "User: 你叫什么？",
        "Assistant: 我是小贱贱！",
    ]

    can_flush: bool = Field(
        default=False,
        description="判断是否接受大模型输出，在判断存在停止词后停止接收和输出"
    )

    # 添加用户输入
    def add_user(self, msg: str):
        self.data.append("User: "+msg)

    # 添加助手输入注意起始部分没有空格
    def add_assistant(self, msg: str):
        self.data.append("Assistant:"+msg)

    # 开始接收模型输出
    def begin_flush(self):
        self.add_assistant("")
        self.can_flush = True

    # 停止接收模型输出
    def end_flush(self):
        self.can_flush = False
        print("\n", end="", flush=True)

    # 接收模型输出
    def flush_assistant(self, msg: str):
        if self.can_flush is True:
            last_msg = self.data.pop()
            last_msg += msg
            for stop in utils.default_stop:
                if stop in last_msg:
                    self.end_flush()
                    self.data.append(last_msg.split(stop)[0])
                    return
            self.data.append(last_msg)

    # 打印在终端
    def print(self, msg: str):
        self.flush_assistant(msg)
        if self.can_flush is True:
            print(msg, end="", flush=True)

    # 转化成模型接收的prompt
    def prompt(self) -> str:
        return "\n\n".join(self.data)

    # 测试时候打印
    def log(self):
        print("\nprompt is :"+self.prompt())


@lru_cache
def get_context():
    return Context()


context = get_context()
