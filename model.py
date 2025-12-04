
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model,prepare_model_for_kbit_training

#定义模型类
class Qwen_NER_LoRA(nn.Module):
    #模型类的初始化方法
    def __init__(self,config):
        super().__init__()#子类调用父类nn.Module的初始化方法
        self.config = config
         # --------------------------------------------------------
        load_in_4bit = config.load_in_4bit
        if load_in_4bit:
            compute_dtype_str = getattr(config, "bnb_4bit_compute_dtype", "bfloat16")
            if compute_dtype_str == "float16":
                compute_dtype = torch.float16
            elif compute_dtype_str == "bfloat16":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = torch.bfloat16   # 默认用 bf16，比较稳
            #BitsAndBytesConfig 里面的字段专门为 4bit/8bit 量化模型准备，告诉基座模型要怎么被量化、怎么存在显存里、怎么在计算时反量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,     #把模型权重量化为 4bit 格式加载
                bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant, #双量化，除了把权重量化为4bit，还把scale缩放因子量化为更小bit。缩放因为也会被存储，scale决定了把浮点数量化到离散整数的精度
                bnb_4bit_quant_type=config.bnb_4bit_quant_type,             #量化类型（"nf4"、"fp4"） "nf4"是QLoRA推荐的 4bit 格式；
                bnb_4bit_compute_dtype=compute_dtype,       #训练和推理的计算梯度损失时用什么精度（bfloat16 or float16看显卡）
            ) 
            # 加载带有完整头部的Qwen底座模型到合适的GPU上  (因果语言模型)
            base_model_1 = AutoModelForCausalLM.from_pretrained(config.model_name_or_path,
                                                              device_map="auto",
                                                              quantization_config=bnb_config,#QLoRA配置
                                                              trust_remote_code=True)
            #（无关LoRA，只轻量化基座）工作：1、保持LayerNorm/lm_head的高精度-更稳定。2、冻结不参与训练的底座权重。
            base_model = prepare_model_for_kbit_training(base_model_1)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path,
                                                              device_map="auto",
                                                              torch_dtype=torch.float16,#LoRA配置
                                                              trust_remote_code=True,)
        #配置LoRA 适配层，并配置相关参数
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  #本项目微调的是因果语言模型
            r           = config.lora_r,
            lora_alpha  = config.lora_alpha,
            lora_dropout= config.lora_dropout,
            bias="none",
            target_modules=config.target_modules,
            )
        # 省显存
        base_model.enable_input_require_grads()                # 帮模型把输入端的梯度开关打开（配合 gradient checkpointing 使用）
        base_model.gradient_checkpointing_enable()             # 开启梯度检查点，显存↓ 时间↑
        base_model.config.use_cache = False                    # 禁用 KV cache
        # (只加LoRA,不动基座) 使用get_peft_model方法在Qwen底座模型（因果语言模型）上插入LoRA适配层。关注可训练参数
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()  # 检查只有 LoRA 在训练
        # 自定义损失函数，忽略label=-100的位置
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-100)

    #前向传播
    def forward(self, input_ids, attention_mask=None, labels=None): 
        outputs = self.model( input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        return {
            "logits": logits,#包含了用来得到预测标签的全部信息
            "labels": labels,#真实标签不参与前向传播
        }
    
    #计算损失接口
    def loss(self, outputs):
        logits = outputs["logits"]   # 是每个位置的预测分布（还没softmax） (batch内第几条样本，序列里第几个token，为每个token打分)
        labels = outputs["labels"]   # 是“真实的下一个token的id”          (batch内第几条样本，序列里第几个token)
        #保证labels不空且是long类型
        if labels is None:
            raise ValueError("labels is None, cannot compute loss.")
        if labels.dtype != torch.long:
            labels = labels.long()
        #为了实现因果模型的“用前一个去预测后一个”，就是用第t个logits的输出分数，去预测的是对应第t+1个的label.
        shift_logits = logits[..., :-1, :].contiguous()  # (1[预测2]，2[预测3]，3[预测4]，4[预测5]) 这里的4位置上预测的5根本不存在，右移截掉最后一位
        shift_labels = labels[..., 1:].contiguous()      # (1，2，3，4) 因为logits第1位预测的是2，而labels的第1没有被预测，左移截掉第一位，自此对齐了
        #contiguous()的作用是保证切片后 底层内存布局的连续，确保之后的重塑形状.view()可以安全使用
        #重塑维度送入交叉熵损失函数计算
        loss = self.loss_ce(
            shift_logits.view(-1, shift_logits.size(-1)), #把前俩维度合并，第三维度保留，重塑为二维
            shift_labels.view(-1),                        #把两维合并为一维
        )
        return loss
    
    #推理：生成式预测实体(使用model里的generate方法)
    @torch.no_grad()
    def predict(self, input_ids, attention_mask=None, **gen_kwargs):
        gen_kwargs.setdefault("max_new_tokens", getattr(self.config, "max_new_tokens", 128))
        gen_kwargs.setdefault("do_sample", False)
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    
