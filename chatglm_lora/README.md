此文件适用于以chatglm为基准模型进行lora训练的情况
# 数据预处理
数据集形式be like：
![image](https://github.com/super-wuliao/LoRA-ChatGLM-Chinese-Alpaca/blob/main/chatglm_lora/c79bd6d1d5096b3067f3b58e76c45ff.png)
转化微调训练数据集（以alpaca数据集为例）为jsonl

    python cover_alpaca2jsonl.py --data_path data/alpaca_data.json --save_path data/alpaca_data.jsonl 
	
分词

    python tokenize_dataset_rows.py --jsonl_path data/alpaca_data.jsonl --save_path data/alpaca --max_seq_length 200 --skip_overlength

--jsonl_path 微调的数据路径, 格式jsonl, 对每行的['context']和['target']字段进行encode

--save_path 输出路径

--max_seq_length 样本的最大长度
# 训练
直接运行下面python代码即可，下载速度慢的话可以手动下载模型，并从[THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)上下载所有的模型文件放到chatglm-6b文件夹中，修改finetune.py文件中model和tokenizer的路径：

    tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("./chatglm-6b", load_in_8bit=True, trust_remote_code=True, device_map="auto")
    
训练：

    python finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output
