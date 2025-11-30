import json

#计算【准确率】【召回率】【F1值】
def indicator(pre_entities, true_entities):
    P_len = 0  # ！存放累加的数值必须初始化，另外.extend()和.append()方法以及字典键值都需要初始化
    R_len = 0
    U_len = 0
    for  batch_pre,barch_truth in zip(pre_entities, true_entities):
        for seq_pre,seq_truth in zip(batch_pre,barch_truth):
            # 少了个if
            P = set(seq_pre)
            R = set(seq_truth)
            U = P & R #求预测标签总数、真实标签总数、预测标签和真实标签的交集总数（预测出的标签数）
            # .extend()将多层嵌套的序列合并为单层列表,允许重复,适合严格逐样本匹配；.=set()​​仅统计当前样本的标签,自动去重，适合全局累加统计
            P_len += len(P)
            R_len += len(R)
            U_len += len(U)
    #计算三个指标
    Precision=U_len/P_len if P_len>0 else 0
    Recall=U_len/R_len if R_len>0 else 0
    F1_score=2*Precision*Recall/(Precision+Recall) if (Precision+Recall)>0 else 0
    return Precision,Recall,F1_score


#把output（字符串）里的实体抽取出来，组成列表{}返回
def parse_entities(text):
    ents = []         #初始空列表用于存抽取出来的实体
    if not text:      #text为空则返回空列表 
        return ents
    for line in text.splitlines(): #把整段文本按“行”切开，遍历每行对去掉每行的两侧空格和换行符
        line = line.strip()   
        if not line or "entity_text" not in line: #如果line为空或者line内没有"entity_text"，则跳过
            continue  
        try:  #尝试解析这一line字符，如果能解析并且是一个dict，而且有"entity_text"，就把这个值加入列表ents{}中
            obj = json.loads(line)
            if isinstance(obj, dict) and "entity_text" in obj:
                ents.append(obj["entity_text"])
        except Exception:#如果json.loads(line)抛异常，就直接跳过
            continue
    return ents


#加载json文件（一次性把整个JSON结构读进来）
def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data
def load_sft_jsonl(path):#（文件不是一个大数组，而是每行一个独立的JSON对象）
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data