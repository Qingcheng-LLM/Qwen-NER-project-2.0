# python -m main 
#gg config -w subscription=''​
#git add -A
#git commit -m "update"
#git push origin main 
import os
import torch
import math
import swanlab
import argparse, re 
from train import train
from valid import valid
import bitsandbytes as bnb
from Dataset import NERDataset
from my_config import my_Config
from model import  Qwen_NER_LoRA
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tool import load_json,load_sft_jsonl
from prepare_date import json_format_transfer
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer#​HuggingFace提供的批处理函数用于告诉DataLoader如何把多个样本合并成一个 batch。1、动态计算每 batch中最长的序列长度；2、对input_ids、attention_mask、labels做填充；3、保证 padding 部分的 label 为 -100（不参与 loss 计算）；4、最终返回可直接输入模型的 batch 张量
def sanitize(s): return re.sub(r'[^a-zA-Z0-9._-]+', '_', s) #将字符串中所有非字母、数字、点、下划线、短横线的字符替换为下划线

def main():
    #-----------------------------------------------argparse/配置------------------------------------------------#
    parser = argparse.ArgumentParser()
    #为参数解析器添加配置文件路径和模型名称的命令行参数
    parser.add_argument("--config", type=str, default="Qwen_NER_LoRA_configs/my_config.json",help="配置文件路径：Qwen_NER_LoRA_configs/my_config.json  保留命令解析以备后续扩展")
    parser.add_argument("--model", type=str, default=None,help="默认结构配置文件里的模型选择，保留命令解析以备后续扩展")#默认不覆盖配置文件中的模型设置
    args = parser.parse_args()#执行实际的命令行参数解析
    config_dict = load_json(args.config)#加载结构配置参数文件，返回包含配置参数的字典
    if args.model:  # 命令行覆盖模型
        config_dict["model_name_or_path"] = args.model
    config = my_Config(**config_dict)#导入结构配置参数，将字典中的键值对作为关键字参数传递给my_Config类的构造函数，创建配置对象
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}  Device: {device}")
    print('loading corpus')
    #-----------------------------------------------准备数据------------------------------------------------#
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # 获取项目根目录
    # 获取转换前的json文件路径
    data_dir_old = os.path.join(BASE_DIR, config.data_dir_old)    
    train_file_path     =  os.path.join(data_dir_old, config.train_file)
    dev_file_path       =  os.path.join(data_dir_old, config.dev_file)
    # 获取转换后的json文件路径
    data_dir_new = os.path.join(BASE_DIR, config.data_dir_new)          
    train_file_new_path =  os.path.join(data_dir_new, config.train_json_new)
    dev_file_new_path   =  os.path.join(data_dir_new, config.dev_jsonn_new)
    # 加载数据进行转换，并加载转换后的新数据
    examples_train = load_json (train_file_path)
    examples_dev   = load_json  (dev_file_path)
    json_format_transfer(examples_train,train_file_new_path)
    json_format_transfer(examples_dev,dev_file_new_path)
    train_data = load_sft_jsonl(train_file_new_path)
    dev_data   = load_sft_jsonl(dev_file_new_path)
    #加载Tokenizer、创建数据集、数据加载器
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=False,trust_remote_code=True,)#不要fast版，因不稳定，信任仓库自带的tokenizer
    train_dataset = NERDataset ( tokenizer, train_data, max_len=config.max_len, is_train=True )
    dev_dataset   = NERDataset ( tokenizer, dev_data,max_len=config.max_len, is_train=False )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                               collate_fn=train_dataset.ner_collate_fn, 
                               num_workers=0,pin_memory=True)
    dev_loader = DataLoader(dev_dataset,batch_size=config.batch_size,shuffle=False,
                            collate_fn=dev_dataset.ner_collate_fn,
                            num_workers=0,pin_memory=True,)
    #----------------------------------------------训练前准备-----------------------------------------------#
    # 初始化模型、优化器、学习率调度器
    model = Qwen_NER_LoRA(config)
    # 只优化需要训练的参数（LoRA 参数），兼容性更好
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=config.learning_rate) #定义优化器
    steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)# 读取梯度累积步数（如果配置里没写，就默认 1）
    total_steps = steps_per_epoch * config.num_epochs       #按批次计算总步数，因为get_linear_schedule_with_warmup 是按 batch 的设计
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    scaler = GradScaler( device="cuda",enabled=(device.type == "cuda"))
    # 创建结果保存目录
    ds  = sanitize(config.dataset_name)
    mdl = sanitize(config.model_name_or_path.split("/")[-1])
    config.target_dir = os.path.join(BASE_DIR, "result", f"{ds}_{mdl}")
    os.makedirs(config.target_dir, exist_ok=True)  # 创建保存目录
    run = swanlab.init(
        project=config.project_name,
        experiment_name=f"{ds}_{mdl}",
        config={
            "model_name_or_path": config.model_name_or_path,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "warmup_ratio": config.warmup_ratio,
            "lora_r": getattr(config, "lora_r", None),
            "lora_alpha": getattr(config, "lora_alpha", None),
            "lora_dropout": getattr(config, "lora_dropout", None),
        }
    )
    print("\n", 20 * "=", f"Training  Qwen_NER_LoRA model on device: {device}", 20 * "=", "\n")
    # 初始化训练状态
    best_score = -1             # 以初始 F1 作为当前最好
    patience_counter = 0                       # 早停计数器，记录指标未提升的epoch数
    #------------------------------------------按轮次训练and验证--------------------------------------------#
    #遍历每个轮次epoch
    for epoch in range(config.num_epochs): 
        print("* Training epoch {}:".format(epoch+1))
        # 调用train()函数完成一个epoch的训练，返回「本轮次耗时」and「损失」。并打印
        epoch_time, epoch_loss = train( model,train_loader,optimizer,config.max_grad_norm,writer=None,epoch=epoch,scheduler=scheduler,accumulation_steps=config.gradient_accumulation_steps, scaler=scaler,use_amp=True, )
        print("-> Training time: {:.4f}s, loss = {:.4f}".format(epoch_time, epoch_loss))
        current_lr = optimizer.param_groups[0]["lr"]
        # 记录训练指标到 SwanLab
        swanlab.log({
            "epoch":       epoch + 1,
            "train_loss":  epoch_loss,
            "train_time":  epoch_time,
            "learning_rate": float(current_lr),
        })
        # 调用valid()函数完成一个epoch的验证，「返回耗时」and「损失」and「准确率、召回率、F1」。并打印
        epoch_time_val, valid_estimator = valid(model,tokenizer,dev_loader)
        print("-> Valid time: {:.4f}s, Precision: {:.4f}%, recall: {:.4f}%, F1: {:.4f}%".format(epoch_time_val, (valid_estimator[0] * 100), (valid_estimator[1] * 100), (valid_estimator[2] * 100)))
        # 记录验证指标到 SwanLab
        swanlab.log({
            "epoch":         epoch + 1,
            "val_precision": valid_estimator[0],
            "val_recall":    valid_estimator[1],
            "val_f1":        valid_estimator[2],
            "val_time":      epoch_time_val,
        })
        # 根据准确率判断是否更好。
        if valid_estimator[2] > best_score:
            best_score = valid_estimator[2]
            patience_counter = 0
            best_ckpt = {
                "model_state_dict": model.state_dict(),
                "config_dict":      config.__dict__,
                "epoch":            epoch,
                "best_score":       best_score,
            }
            torch.save(best_ckpt, os.path.join(config.target_dir, "checkpoint.pth"))
        else:
            patience_counter += 1
            # 早停触发
            if patience_counter >= config.patience:
                print("-> Early stopping: patience limit reached, stopping...")
                break
    swanlab.finish()
    print("训练完成！使用以下命令查看")

if __name__ == "__main__":
    main()