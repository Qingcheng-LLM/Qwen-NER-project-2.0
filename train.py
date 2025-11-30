
import time
from tqdm.auto import tqdm
import torch.nn as nn

#训练模型
def train(model,dataloader,optimizer,max_gradient_norm, writer=None, epoch=None, scheduler=None, accumulation_steps: int = 1):
    model.train()             #开启训练模式
    device = next(model.parameters()).device  #自动获取模型所在设备
    epoch_start =time.time()  #记录epoch开始时间
    optimizer.zero_grad()     #清空梯度
    batch_time_avg = 0.0      #累计批次处理时间
    running_loss = 0.0        #累计损失值
    #1、遍历数据
    tqdm_batch_iterator = tqdm(dataloader)                    #加载数据并显示批次进度条
    for batch_index, batch in enumerate(tqdm_batch_iterator): #加载每批次数据及批次的索引
        batch_start = time.time()               #记录batch开始时间
    #2、数据移动到设备
        #从字典中获取数据
        inputs = batch['input_ids'].to(device)
        att_masks = batch['attention_mask'].to(device)
        label = batch['labels'].to(device)
    #3、前向传播  计算损失
        
        output = model(inputs, att_masks, label) #输入输入数据 and掩码 and标签「进行前向传播」
        loss = model.loss(output)   
        # 梯度累积：先把 loss 均摊到每个小 step
        loss = loss / accumulation_steps
        loss.backward()                          #根据损失计算的梯度来「反向传播」
    #4、每 accumulation_steps 个 batch 再更新一次参数
        if (batch_index + 1) % accumulation_steps == 0 or (batch_index + 1) == len(dataloader):
            nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm) #「梯度裁剪」来防止梯度爆炸
            optimizer.step()                                                #根据反向传播计算新梯度来「更新参数」
            optimizer.zero_grad()                                           # 清空梯度
            if scheduler is not None:
                scheduler.step()  #更新学习率
        #5、记录该批次训练指标
            batch_time = time.time() - batch_start   #累计每个批次的训练时间
            batch_time_avg += batch_time
            running_loss += loss.item()                   #累计损失值
            #记录每批次的损失到TensorBoard
            if writer is not None and epoch is not None:
                global_step = epoch * len(dataloader) + batch_index
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        #6、更新进度条描述
            description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" .format(batch_time_avg / (batch_index + 1),running_loss / (batch_index + 1))
            tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start            #记录该轮次的训练时间
    epoch_loss = running_loss / len(dataloader)       #计算平均损失
    return epoch_time, epoch_loss