# LoRA-ChatGLM-Chinese-Alpaca
  为了促进大模型在中文NLP社区的开放研究以及广大垂直领域的应用，
本项目使用LoRA对基于清华[ChatGLM](https://github.com/THUDM/ChatGLM-6B)（中英双语训练）以及其他大佬使用[中文指令精调的Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca)进行进一步的微调，并给出训练得到的LoRA权重及与原模型参数融合的方式，各位可以根据需求自取。

针对ChatGLM模型，因为其本身预训练的语料就包含了大量中文，已经能生成相当符合人类偏好的回答，因此对其进行进一步的通用中文指令精调意义不大，
（这里给出我使用五万BELLE项目产生的中文指令数据集及在此基础上训练七个epoch后得到的LoRA权重，各位可自行对比与原ChatGLM的区别）
后续会直接尝试基于LoRA、P-tuning v2等参数有效性方法针对垂直领域进行训练。（当然首先是中文语法纠错领域，既然ChatGPT让我无路可走，
那么本着打不过就加入的原则我得抓紧提升语法纠错的效果，卷死之前传统方法的同行们）

由于LLaMA模型并不具备类ChatGPT直接对话的能力，
后来斯坦福老哥使用5万左右的数据（ChatGPT生成的答案，实乃用魔法打败魔法，只有openai受伤的世界is coming）对LLaMA进行了finetune，发布了Alpaca（小羊驼），
使其能够支持像ChatGPT一样进行对话。同时由于其训练语料基本为英语，看到有人直接使用LoRA进行中文语料微调的效果并不是很理想，因此这里使用
[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
开源的使用中文纯文本数据对LLaMA进行二次预训练并使用指令数据精调的Chinese-Alpaca（中国小羊驼）作为基准模型，继续使用LoRA针对垂直领域进行训练。

# LoRA方法
LoRA，英文全称[Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)，直译为大语言模型的低阶适应，是一种PEFT（参数高效性微调方法），
而低代价微调大模型的方法有很多，比如p-tuning，p-tuning v2，prefix tuning，prompt tuning，添加adapter模块等，各位有兴趣可自行查阅。

而选择LoRA的原因是其相对来说更加通用，可当成模型插件即插即用，如可基于ChatGLM一个模型训练文本摘要、语法纠错等多个LoRA模型
（占内存很小，6B大小的大模型仅需几十兆的大小保存LoRA权重），因此使用及训练起来很方便。而且相对于引入adapter模块来说，使用LoRA推理没有引入新的时延，
与使用原模型相比仅多了一个加法操作即将原模型参数与LoRA参数相加，因此不会导致模型推理变慢。（总之不只是因为有现成的代码就对了）


# LoRA微调
这一部分主要参考[ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning)的代码，由于使用了两个基准模型，因此相关代码分别放到了chatglm_lora和chinese_alpaca_lora两个文件夹中


# LoRA模型融合
这一部分主要参考[Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)的代码

# 中文指令数据集
如果各位觉得上述的基准模型ChatGLM和Chinese-Alpaca的中文问题的生成效果不太好的话，可以自行使用中文指令数据集进行LoRA微调，之后再模型融合就可以继续垂直领域的模型训练了。

下面是一些开源的中文指令数据集：

[包含约100万条由BELLE项目生成的中文指令数据](https://huggingface.co/datasets/BelleGroup/train_1M_CN)

[alpaca中文指令微调数据集,机翻及self-instruct生成](https://github.com/carbonz0/alpaca-chinese-dataset)

[alpaca机翻人工校验并加入了新的中文聊天对话](https://github.com/hikariming/alpaca_chinese_dataset)

[收集了23个常见的中文数据集，对于每个任务，由人工书写若干种指令模板](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)

[GuanacoDataset 多语言指令数据集](https://huggingface.co/datasets/JosephusCheung/GuanacoDataset)




