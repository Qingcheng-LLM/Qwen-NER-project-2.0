
import time
import torch
from tqdm.auto import tqdm
from tool import indicator, parse_entities

def valid(model, tokenizer, dev_loader):
    model.eval()      #开启评估模式（在此会关闭dropout等训练专用层）
    device = next(model.parameters()).device #直接从模型参数获取设备
    start_time = time.time()
    #准备传给 indicator 的输入格式：[[样本1实体列表, 样本2实体列表, ...]]
    all_pred = [[]]
    all_true = [[]]
    #以下过程不计算梯度
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Valid (generate)"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_outputs = batch["gold_outputs"]  # 真实答案文本列表
            #调用模型封装好的 predict 进行预测（里面已 @torch.no_grad）
            gen_ids = model.predict(input_ids=input_ids, attention_mask=attention_mask)
            #生成结果是 [prompt+新生成内容] 的token序列，需要切掉前面prompt的长度，只保留回答
            prompt_len = input_ids.size(1)
            gen_only = gen_ids[:, prompt_len:]
            #解码成字符串
            pred_texts = tokenizer.batch_decode(gen_only,skip_special_tokens=True)
            #解析实体 & 累计指标
            for pred_text, gold_output in zip(pred_texts, gold_outputs):
                pred_ents = parse_entities(pred_text)
                true_ents = parse_entities(gold_output)
                all_pred[0].append(pred_ents)
                all_true[0].append(true_ents)
    # indicator 计算 P / R / F1
    Precision, Recall, F1 = indicator(all_pred, all_true)
    estimator = (Precision, Recall, F1)
    epoch_time = time.time() - start_time
    return epoch_time, estimator

"""

这里需要dev_dataset数据【原始指令样本列表，每个元素为 {instruction,input,output}】
因为我们想要的是：1、拿 instruction + input + gold的K 自己拼一个 提示prompt；
2、用模型生成答案；
3、然后与 gold的V 做对比。（gold指的是：数据集中本来就带的“标准答案/正确标注”的输出）

而dev_loader数据提供的是已经 tokenized + padding + labels 的 batch，
此时已经经过了dev_dataset.ner_collate_fn自定义函数的处理，
它的input_ids(instruction+input+gold)里面往往已经包含了答案部分,
不适合用来做“只给 prompt、看模型生成什么”的评估。

训练时：input_ids= sys + usr + ass + answer；
验证生成时：只给 sys + usr + ass，让模型自己“续写” answer 部分。

"""