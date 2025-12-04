
import torch
from torch.utils.data import Dataset  

class NERDataset(Dataset):
    ## 类初始化
    def __init__(self, tokenizer,data,max_len=None,is_train=True):
        self.data = data
        self.tokenizer=tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.encoded_data = []
        for ex in data:
            if is_train:
                encoded = self.encode_one(ex)
            else:
                encoded = self.encode_one_valid(ex)
            self.encoded_data.append(encoded)
    def __len__(self):#返回数据集总长度，方便dataloader确定迭代的次数
        return len(self.encoded_data)
    def __getitem__(self, idx):#可以通过索引直接访问数据集中任意一条数据
        return self.encoded_data[idx]

    ## 编码单条数据为模型可以接受的格式
    def encode_one(self,example):
        input = example["instruction"] +"\n"+  example["input"]
        output = example["output"]#答案部分
        #变成 token id          qwen的结束符是<|endoftext|>
        inputs_tokens =self.tokenizer(input,add_special_tokens=False) # 把system+user+assistant开头这一整段编码成token序列
        answer_tokens =self.tokenizer(output,add_special_tokens=False) # 把答案编码成token序列///add_special_token=False:不要再自动加 BOS/EOS 之类的东西
        eos_ids = self.tokenizer(self.tokenizer.eos_token)["input_ids"]
        #构造最终的input_ids\attention_mask\label ----------（对话格式+回答+<eos>
        input_ids      = inputs_tokens["input_ids"] + answer_tokens["input_ids"] + eos_ids 
        attention_mask = [1] * len(input_ids)
        labels         = [-100] * len(inputs_tokens["input_ids"]) + answer_tokens["input_ids"] + eos_ids
        #超长截断
        input_ids      = input_ids     [:self.max_len]
        attention_mask = attention_mask[:self.max_len] 
        labels         = labels        [:self.max_len]
        return {"input_ids": input_ids,"attention_mask": attention_mask,"labels": labels}
    
    def encode_one_valid(self,example):
        input = example["instruction"] +"\n"+  example["input"]+ "\n"
        output = example["output"]#答案部分
        #变成 token id   
        inputs_tokens =self.tokenizer(input,add_special_tokens=False) # 把system+user+assistant开头这一整段编码成token序列
        eos_ids = self.tokenizer(self.tokenizer.eos_token)["input_ids"]
        #构造最终的input_ids\attention_mask ----------（对话格式+<eos>
        input_ids      = inputs_tokens["input_ids"]
        attention_mask = [1] * len(input_ids)
        labels         = [-100] * len(input_ids)  # 保持维度统一,验证时不需要标签
        #超长截断
        input_ids      = input_ids     [:self.max_len]
        attention_mask = attention_mask[:self.max_len] 
        labels         = labels        [:self.max_len]
        return {"input_ids": input_ids,"attention_mask": attention_mask,"labels": labels,"gold_output": output}
    

    ## 按批次动态填充
    def ner_collate_fn(self,batch):
        batch_input_ids = [ex["input_ids"] for ex in batch]
        batch_attention_mask = [ex["attention_mask"] for ex in batch]
        batch_labels = [ex["labels"] for ex in batch]
        #取每个批次的最大序列长度
        max_len = max(len(seq) for seq in batch_input_ids) 
        set_input_ids = []
        set_attention_mask = []
        set_labels = []
        #对批次内的每个序列的三类数据都进行填充
        for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_len - len(input_ids)# 计算每个序列数据需要填充的长度
            if self.is_train:
                # 训练时右填充
                padded_input_ids = input_ids + [self.tokenizer.eos_token_id] * pad_len# 填充每个序列的token
                padded_attention = attention_mask + [0] * pad_len# 填充attention_mask
                padded_labels = labels + [-100] * pad_len# 填充labels（用-100忽略）
            else:
                # 验证/测试时左填充
                padded_input_ids = [self.tokenizer.eos_token_id] * pad_len + input_ids
                padded_attention = [0] * pad_len + attention_mask
                padded_labels = [-100] * pad_len + labels
            #填充以后收集本批次填充好的数据
            set_input_ids.append(padded_input_ids)
            set_attention_mask.append(padded_attention)
            set_labels.append(padded_labels)
        batch_dict= {
            'input_ids': torch.tensor(set_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(set_attention_mask, dtype=torch.long),
            'labels': torch.tensor(set_labels, dtype=torch.long)
            }
        if not self.is_train:
            batch_dict['gold_outputs'] = [ex["gold_output"] for ex in batch]
        return batch_dict

