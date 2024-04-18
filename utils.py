# 停止标志，和PIPELINE_ARGS里的停止标志不一样，在对话中无法完全保证模型输出停止符号就是结束符号，不过可以在生成下段的时候采用截取的方式控制输出
default_stop = [
    "\n\n",
    "\n\nUser",
    "\n\nQuestion",
    "\n\nQ",
    "\n\nHuman",
    "\n\nBob",
    "\n\nAssistant",
    "\n\nAnswer",
    "\n\nA",
    "\n\nBot",
    "\n\nAlice",
]