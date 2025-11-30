
## 🚀项目简介:
这里一个基于Pytorch框架和Hugging Face生态，使用Qwen2.5-7B作为基座模型，使用LoRA或QLoRA进行微调，完成的一个命名实体识别的任务，旨在更好的理解指令微调原理。项目包括以下模块：<br>  
 1.数据准备与预处理<br>
 2.模型设计与结构配置类<br>
 3.模型训练与模型评估，并提供可视化（Swanlab）<br> 

项目里提供Qwen2.5-7B模型权重，并提供了下载模型的脚本（downloadmodel.py），首次运行需要下载预训练模型，后续可直接运行，下载的预训练模型均存储在LLM_model文件夹里。<br>  
项目提供bc2gm数据集，所有可见配置的超参数在Qwen_NER_LoRA_configs文件夹下。<br>  
项目默认为LoRA微调，如果用QLoRA，可以在my_config.json文件里把"load_in_4bit"改为True<br>  
在保存项目运行结果时，会连同所选模型运行结果、超参数配置文件、项目运行最好的指标一同保存，方便后续复现表现较为出色结果，结果存储在result文件夹下。<br>  
在项目main.py文件开头已经给出项目执行命令，配好环境后，可直接复制命令行终端运行，运行结果的F1值约为84%。<br>  

    
## 📊 数据集来源： 

 1.bc2gm命名实体识别数据集。下载地址：https://github.com/spyysalo/bc2gm-corpus?utm_source=chatgpt.com<br>  

## 📊 预训练模型：
 预训练模型下载脚本已在LLM_model文件夹下给出。<br>  
 1.Qwen2.5-7B <br>

## 🗓项目目录: 
```
Qwen-NER-LoRA/  
├─ Qwen-NER-LoRA_configs/       # 项目配置  
├─ dataset/                     # 数据集（bc2gm）  
├─ LLM_model/                   # 预训练模型  
├─ result/                      # 结果与日志   
├─ Dataset.py                   # 数据集读取  
├─ prepare_data.py              # 准备数据集  
├─ model.py                     # 模型结构  
├─ train.py / valid.py          # 训练与验证入口  
├─ main.py                      # 主入口  
├─ downloadmodel.py             # 预训练权重下载脚本  
└─ my_config.py / tool.py       # 配置与工具函数  
```